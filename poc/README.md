# TensorFlow Lite From-Source Static Compilation POC

This directory contains the original Proof of Concept (POC) demonstrating how to compile TensorFlow Lite C API from source with CMake, consolidate its static library dependencies using an `ar` MRI script, and produce a fully static Go binary.

> **Note**: For day-to-day development or consuming `go-tflite`, you do **not** need this folder. The root library already ships with precompiled binaries and headers so you can simply `go get` it. This folder is preserved for reproducibility and reference.

## Building and Running the POC

```bash
docker build -t tflite-poc:test poc/
docker run --rm tflite-poc:test
```
