/// <reference types="@webgpu/types" />
import { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
  Processor,
  PreTrainedModel,
} from "@huggingface/transformers";
import "./App.css";
import JSZip from "jszip";
import { saveAs } from "file-saver";



export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<any>(null);
  const modelRef = useRef<PreTrainedModel | null>(null);
  const processorRef = useRef<Processor | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalSize, setTotalSize] = useState(0);

  useEffect(() => {
    (async () => {
      try {
        if (!navigator?.gpu) {
          throw new Error("WebGPU is not supported in this browser.");
        }
      
        window.fetch = new Proxy(window.fetch, {
          async apply(target, thisArg, args) {
            if (!args[0].endsWith('.onnx')) {
              return Reflect.apply(target, thisArg, args);
            }

            // Call the original fetch
            const response = await Reflect.apply(target, thisArg, args);

            // Clone the response so we can read the body
            const contentLength = response.headers.get('content-length');
            const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
            setTotalSize(totalBytes / 1024 / 1024);

            let downloadedBytes = 0;

            // If there's no body, just return the response as is
            if (!response.body) {
              return response;
            }

            // Create a new readable stream to read the response body
            const reader = response.body.getReader();

            // Create a new response with the stream that tracks progress
            const stream = new ReadableStream({
              async pull(controller) {
                // Read the data
                const { done, value } = await reader.read();

                // If we're done, close the stream
                if (done) {
                  controller.close();
                  return;
                }

                // Track the downloaded bytes
                downloadedBytes += value.length;

                // Log the progress
                if (totalBytes) {
                  setProgress((prevProgress) => {
                    const progress = (downloadedBytes / totalBytes) * 100;
                    return Math.max(prevProgress, progress);
                  });
                }

                // Send the data chunk to the stream
                controller.enqueue(value);
              }
            });

            // Return the new response with the intercepted stream
            return new Response(stream, {
              headers: response.headers,
              status: response.status,
              statusText: response.statusText,
            });
          }
        });

        //const model_id = "Xenova/modnet";
        const model_id = "briaai/RMBG-1.4";
        if (env.backends.onnx.wasm) {
          env.backends.onnx.wasm.proxy = false;
          // Default is https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.14/dist/ort-wasm-simd-threaded.jsep.wasm
          // But sometimes it's slow, about 21.5mb
          env.backends.onnx.wasm.wasmPaths = "https://unpkg.com/@huggingface/transformers@3.0.0-alpha.14/dist/";
        }
        modelRef.current ??= await AutoModel.from_pretrained(model_id, {
          device: "webgpu",
          dtype: "fp16",
        });
        processorRef.current ??= await AutoProcessor.from_pretrained(model_id);
      } catch (err) {
        setError(err);
      }
      // Still needs to wait "ort-wasm-simd-threaded.jsep.wasm" to be downloaded...
      setIsLoading(false);
    })();
  }, []);

  

  function Err() {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-4xl mb-2">ERROR</h2>
          <p className="text-xl max-w-[500px]">{error.message}</p>
        </div>
      </div>
    );
  }

  function Loading() {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center">
        <p className="text-xl font-semibold text-gray-700">Loading background removal model(briaai/RMBG-1.4)...</p>
        <p className="text-sm text-gray-500 mt-2">Total size: {totalSize.toFixed()} MB.This may take a few moments, please wait.</p>
        <div className="w-2/3 bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mt-4">
          <div className="bg-blue-600 h-2.5 rounded-full animate-pulse" style={{ width: `${progress.toFixed()}%` }} />
        </div>
      </div>
    );
  }

  function App() {
    const [images, setImages] = useState<string[]>([]);
    const [processedImages, setProcessedImages] = useState<string[]>([]);
    const [isProcessing, setIsProcessing] = useState<boolean>(false);
    const [isDownloadReady, setIsDownloadReady] = useState<boolean>(false);
    const onDrop = useCallback((acceptedFiles: File[]) => {
      setImages((prevImages) => [
        ...prevImages,
        ...acceptedFiles.map((file) => URL.createObjectURL(file)),
      ]);
    }, []);

    const {
      getRootProps,
      getInputProps,
      isDragActive,
      isDragAccept,
      isDragReject,
    } = useDropzone({
      onDrop,
      accept: {
        "image/*": [".jpeg", ".jpg", ".png"],
      },
    });

    const removeImage = (index: number) => {
      setImages((prevImages) => prevImages.filter((_, i) => i !== index));
      setProcessedImages((prevProcessed) =>
        prevProcessed.filter((_, i) => i !== index),
      );
    };

    const processImages = async () => {
      const model = modelRef.current;
      const processor = processorRef.current;
      if (!model || !processor) {
        return;
      }

      setIsProcessing(true);
      setProcessedImages([]);

      for (let i = 0; i < images.length; ++i) {
        // Load image
        const img = await RawImage.fromURL(images[i]);

        // Pre-process image
        const { pixel_values } = await processor(img);

        // Predict alpha matte
        const { output } = await model({ input: pixel_values });

        const maskData = (
          await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
            img.width,
            img.height,
          )
        ).data;

        // Create new canvas
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          continue;
        }

        // Draw original image output to canvas
        ctx.drawImage(img.toCanvas(), 0, 0);

        // Update alpha channel
        const pixelData = ctx.getImageData(0, 0, img.width, img.height);
        for (let i = 0; i < maskData.length; ++i) {
          pixelData.data[4 * i + 3] = maskData[i];
        }
        ctx.putImageData(pixelData, 0, 0);
        setProcessedImages((prevProcessed) => [
          ...prevProcessed,
          canvas.toDataURL("image/png"),
        ]);
      }

      setIsProcessing(false);
      setIsDownloadReady(true);
    };

    const downloadAsZip = async () => {
      const zip = new JSZip();
      const promises = images.map(
        (image, i) =>
          new Promise((resolve) => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            if (!ctx) {
              resolve(null);
              return;
            }

            const img = new Image();
            img.src = processedImages[i] || image;

            img.onload = () => {
              canvas.width = img.width;
              canvas.height = img.height;
              ctx.drawImage(img, 0, 0);
              canvas.toBlob((blob) => {
                if (blob) {
                  zip.file(`image-${i + 1}.png`, blob);
                }
                resolve(null);
              }, "image/png");
            };
          }),
      );

      await Promise.all(promises);

      const content = await zip.generateAsync({ type: "blob" });
      saveAs(content, "images.zip");
    };

    const clearAll = () => {
      setImages([]);
      setProcessedImages([]);
      setIsDownloadReady(false);
    };

    const copyToClipboard = async (url: string) => {
      try {
        // Fetch the image from the URL and convert it to a Blob
        const response = await fetch(url);
        const blob = await response.blob();

        // Create a clipboard item with the image blob
        const clipboardItem = new ClipboardItem({ [blob.type]: blob });

        // Write the clipboard item to the clipboard
        await navigator.clipboard.write([clipboardItem]);

        console.log("Image copied to clipboard");
      } catch (err) {
        console.error("Failed to copy image: ", err);
      }
    };

    const downloadImage = (url: string) => {
      const link = document.createElement("a");
      link.href = url;
      link.download = "image.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
    return <>
        <div
          {...getRootProps()}
          className={`p-8 my-8 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors duration-300 ease-in-out
            ${isDragAccept ? "border-green-500 bg-green-900/20" : ""}
            ${isDragReject ? "border-red-500 bg-red-900/20" : ""}
            ${isDragActive ? "border-blue-500 bg-blue-900/20" : "hover:border-blue-300 hover:bg-blue-900/10"}
          `}
        >
          <input {...getInputProps()} className="hidden" />
          <p className="text-lg mb-2">
            {isDragActive
              ? "Drop the images here..."
              : "Drag and drop some images here"}
          </p>
          <p className="text-sm text-gray-400">or click to select files</p>
        </div>
        <div className="flex flex-col gap-4 mb-8">
          <button
            onClick={processImages}
            disabled={isProcessing || images.length === 0}
            className="px-6 py-3 text-white rounded-md bg-blue-400 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200 text-lg font-semibold w-36"
          >
            {isProcessing ? "Processing..." : "Process"}
          </button>
          <div className="flex gap-4">
            <button
              onClick={downloadAsZip}
              disabled={!isDownloadReady}
              className="px-3 py-1 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-black disabled:bg-gray-700 disabled:cursor-not-allowed disabled:hidden transition-colors duration-200 text-sm"
            >
              Download as ZIP
            </button>
            <button
              onClick={clearAll}
              disabled={images.length === 0}
              className="px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-black transition-colors duration-200 text-sm disabled:hidden"
            >
              Clear All
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {images.map((src, index) => (
            <div key={index} className="relative group">
              <img
                src={processedImages[index] || src}
                alt={`Image ${index + 1}`}
                className="rounded-lg object-cover w-full h-48"
              />
              {processedImages[index] && (
                <div className="absolute inset-0 bg-black bg-opacity-70 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg flex items-center justify-center">
                  <button
                    onClick={() =>
                      copyToClipboard(processedImages[index] || src)
                    }
                    className="mx-2 px-3 py-1 bg-white text-gray-900 rounded-md hover:bg-gray-200 transition-colors duration-200 text-sm"
                    aria-label={`Copy image ${index + 1} to clipboard`}
                  >
                    Copy
                  </button>
                  <button
                    onClick={() => downloadImage(processedImages[index] || src)}
                    className="mx-2 px-3 py-1 bg-white text-gray-900 rounded-md hover:bg-gray-200 transition-colors duration-200 text-sm"
                    aria-label={`Download image ${index + 1}`}
                  >
                    Download
                  </button>
                </div>
              )}
              <button
                onClick={() => removeImage(index)}
                className="absolute top-2 right-2 bg-black bg-opacity-50 text-white w-6 h-6 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-opacity-70"
                aria-label={`Remove image ${index + 1}`}
              >
                &#x2715;
              </button>
            </div>
          ))}
        </div>
    </>
  }

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-gray-100 px-12 md:px-0">
      <div className="w-full md:w-2/3 bg-white rounded-lg p-8">
        <h1 className="text-4xl font-bold mb-2">
          Background Remover
        </h1>
        <div className="min-h-72">
          {error && <Err />}
          {isLoading && !error && <Loading />}
          {!error && !isLoading && <App />}
        </div>
        <footer className="md:flex gap-y-2 items-center justify-between mt-20 text-gray-400">
          <div className="items-end grid-flow-col ">
            <p>Copyright © 2024 - All right reserved </p>
          </div>
          <div className="flex gap-2">
          {/* add home link to about page */}
            <a href="https://blog.sideeffect.dev/about" target="_blank">
              <svg 
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
                <path d="M2 12h20"/>
              </svg>
            </a>
            <a href="https://github.com/iketiunn/rmbg" target="_blank">
              <svg
                width="20"
                height="20"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 512 512"
                className="inline-block h-5 w-5 fill-current md:h-6 md:w-6"
              >
                <path d="M256,32C132.3,32,32,134.9,32,261.7c0,101.5,64.2,187.5,153.2,217.9a17.56,17.56,0,0,0,3.8.4c8.3,0,11.5-6.1,11.5-11.4,0-5.5-.2-19.9-.3-39.1a102.4,102.4,0,0,1-22.6,2.7c-43.1,0-52.9-33.5-52.9-33.5-10.2-26.5-24.9-33.6-24.9-33.6-19.5-13.7-.1-14.1,1.4-14.1h.1c22.5,2,34.3,23.8,34.3,23.8,11.2,19.6,26.2,25.1,39.6,25.1a63,63,0,0,0,25.6-6c2-14.8,7.8-24.9,14.2-30.7-49.7-5.8-102-25.5-102-113.5,0-25.1,8.7-45.6,23-61.6-2.3-5.8-10-29.2,2.2-60.8a18.64,18.64,0,0,1,5-.5c8.1,0,26.4,3.1,56.6,24.1a208.21,208.21,0,0,1,112.2,0c30.2-21,48.5-24.1,56.6-24.1a18.64,18.64,0,0,1,5,.5c12.2,31.6,4.5,55,2.2,60.8,14.3,16.1,23,36.6,23,61.6,0,88.2-52.4,107.6-102.3,113.3,8,7.1,15.2,21.1,15.2,42.5,0,30.7-.3,55.5-.3,63,0,5.4,3.1,11.5,11.4,11.5a19.35,19.35,0,0,0,4-.4C415.9,449.2,480,363.1,480,261.7,480,134.9,379.7,32,256,32Z"></path>
              </svg>
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}
