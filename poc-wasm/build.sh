#!/bin/bash
set -e
echo "Building TFLite WebAssembly module..."
docker build -t tflite-wasm-builder -f Dockerfile .
echo "Extracting tflite.wasm..."
docker rm -f tflite-wasm-extract 2>/dev/null || true
docker create --name tflite-wasm-extract --entrypoint /tflite.wasm tflite-wasm-builder
docker cp tflite-wasm-extract:/tflite.wasm ./tflite.wasm
docker rm tflite-wasm-extract
echo "Done! tflite.wasm is ready ($(ls -lh tflite.wasm | awk '{print $5}'))."
