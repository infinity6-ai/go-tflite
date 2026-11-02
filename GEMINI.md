# GEMINI.md - go-tflite Developer & Agent Guidance

## 1. Project Overview
`go-tflite` (`github.com/infinity6-ai/go-tflite`) provides an idiomatic Go binding for the **TensorFlow Lite C API** targeting **Linux amd64**.
Its core design goal is **zero recompilation for consumers**: any developer or CI environment can `go get` this module and immediately produce a **100% statically linked standalone Go binary** (`statically linked`, no glibc dynamic runtime requirements, runnable in `FROM scratch` Docker images) using only standard `gcc`.

---

## 2. Architecture & Bundled Artifacts
- **`lib/linux_amd64/libtensorflowlite_c_all.a` (~14 MB)**:
  A monolithic static archive containing `libtensorflowlite_c.a`, `libtensorflow-lite.a`, and all transitive static dependencies (`abseil-cpp`, `ruy`, `flatbuffers`, `cpuinfo`, `farmhash`, `fft2d`, `pthreadpool`, `gemmlowp`).
  Built by combining CMake build outputs using a GNU `ar` MRI script. Fits well within GitHub's 100 MB file limit.
- **`include/` (~150 KB)**:
  Minimal set of stripped C API header files (19 `.h` files). Do **not** commit `.cc`, `.lds`, `.bzl`, `BUILD`, or `CMakeLists.txt` files into this directory.
- **CGO Directives (`cgo.go`)**:
  ```go
  //go:build linux && amd64

  package tflite

  /*
  #cgo linux,amd64 CFLAGS: -I${SRCDIR}/include
  #cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64 -ltensorflowlite_c_all -lstdc++ -lm -lpthread -ldl
  #include <tensorflow/lite/c/c_api.h>
  #include <stdlib.h>
  */
  import "C"
  ```
  `${SRCDIR}` automatically expands to the package directory within Go module caches.

---

## 3. Key Files & Structure
- **`cgo.go`**: CGO build tags and `Version()` function.
- **`model.go`**: Model lifecycle (`NewModelFromFile`, `NewModelFromBuffer`, `Delete`).
- **`interpreter.go`**: Interpreter execution (`NewInterpreter`, `AllocateTensors`, `Invoke`, tensor count & retrieval).
- **`tensor.go`**: Tensor inspection and typed buffer accessors (`CopyFromFloat32`, `Float32s`, `Shape`, `ByteSize`, etc.).
- **`types.go`**: `TensorType` enum mappings matching `TfLiteType`.
- **`unsupported.go`**: Build-tag guard (`//go:build !(linux && amd64)`) that panics if compiled on other platforms.
- **`testdata/add.bin`**: Sample model for testing (adds input float32 values to constant 4.0; input 2.0 -> output 6.0).
- **`examples/static_app/`**: A consumer application demonstrating `//go:embed` model embedding and static compilation.
- **`test.sh`**: The end-to-end verification script for the library and consumer app.
- **`build.sh`**: Project task dispatcher (`clean`, `build`, `go`, `update`, `run`, `codegen`, `test_unit`, `test_remote`, `test`, `release`).
- **`poc/`**: The original from-source POC directory containing `poc/Dockerfile` (used to build TFLite from source and reproduce `libtensorflowlite_c_all.a` if updates are ever needed) and `poc/main.go`.

---

## 4. Workflows & Validation Commands

### Run Unit Tests
Always run unit tests before and after code changes:
```bash
go test -v .
```
Expected execution time: < 0.05s.

### Full Pipeline Verification
Run the root verification script:
```bash
./test.sh
```
This script:
1. Runs `go test -v .`.
2. Compiles `examples/static_app` statically into `target/go-tflite-app`.
3. Verifies `file target/go-tflite-app` reports `statically linked`.
4. Verifies `ldd target/go-tflite-app` reports `not a dynamic executable`.
5. Executes the binary locally on the host.
6. Builds a minimal `FROM scratch` Docker image and executes the binary inside it.

### Consumer Static Build Command
To produce a static binary from Go code using this library:
```bash
CGO_ENABLED=1 go build -ldflags="-linkmode external -extldflags '-static -s -w'" -o <output_binary> .
```

---

## 5. Maintenance: Upgrading TensorFlow Lite
If TensorFlow Lite needs to be rebuilt with a newer version in the future:
1. Update the TensorFlow git tag/branch in `poc/Dockerfile` (currently `v2.16.2`).
2. CMake options must preserve CPU portability:
   `-DTFLITE_C_BUILD_SHARED_LIBS=OFF -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release`
3. The MRI script merges all `.a` files:
   ```bash
   echo "CREATE libtensorflowlite_c_all.a" > combine.mri
   find . -name "*.a" ! -name "libtensorflowlite_c_all.a" | while read -r f; do
       echo "ADDLIB $f" >> combine.mri
   done
   echo "SAVE" >> combine.mri && echo "END" >> combine.mri
   ar -M < combine.mri && ranlib libtensorflowlite_c_all.a
   ```
4. Copy the resulting `.a` file back to `lib/linux_amd64/libtensorflowlite_c_all.a`.
5. Keep only `.h` header files inside `include/`.
