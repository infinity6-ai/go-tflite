# go-tflite

A standalone, statically linkable Go binding for **TensorFlow Lite C API** on **Linux amd64**.

## Why go-tflite?

Standard TensorFlow Lite Go wrappers require users to install complex C++ toolchains, Python, Bazel or CMake, and compile gigabytes of TensorFlow source code just to run inference.

**`go-tflite` solves this:**
- **Zero Recompilation**: Ships with a precompiled, monolithic static archive (`libtensorflowlite_c_all.a`, ~14 MB) and minimal C API headers.
- **Pure Go Workflow**: Anyone can `go get github.com/infinity6-ai/go-tflite` and build their project in seconds.
- **100% Static Linking**: Build completely self-contained Linux amd64 binaries that run on bare metal or inside `FROM scratch` Docker containers without any shared libraries or dynamic linkers.
- **Idiomatic Go API**: Safe, clean abstractions around models, interpreters, and tensors with automatic buffer handling.

---

## Requirements

- **OS/Arch**: Linux `amd64`
- **Compiler**: `gcc` / `build-essential` (standard CGO requirement)
- **Go**: 1.22+

---

## Installation

```bash
go get github.com/infinity6-ai/go-tflite
```

---

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	tflite "github.com/infinity6-ai/go-tflite"
)

func main() {
	// 1. Load model from file (or NewModelFromBuffer)
	model, err := tflite.NewModelFromFile("model.tflite")
	if err != nil {
		log.Fatalf("failed to load model: %v", err)
	}
	defer model.Delete()

	// 2. Create interpreter with options
	interp, err := tflite.NewInterpreter(model, tflite.Options{NumThreads: 2})
	if err != nil {
		log.Fatalf("failed to create interpreter: %v", err)
	}
	defer interp.Delete()

	// 3. Allocate tensors
	if err := interp.AllocateTensors(); err != nil {
		log.Fatalf("failed to allocate tensors: %v", err)
	}

	// 4. Fill input tensor
	inputTensor := interp.GetInputTensor(0)
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	if err := inputTensor.CopyFromFloat32(inputData); err != nil {
		log.Fatalf("failed to copy input: %v", err)
	}

	// 5. Run inference
	if err := interp.Invoke(); err != nil {
		log.Fatalf("failed to invoke: %v", err)
	}

	// 6. Read output tensor
	outputTensor := interp.GetOutputTensor(0)
	results, err := outputTensor.Float32s()
	if err != nil {
		log.Fatalf("failed to read output: %v", err)
	}

	fmt.Println("Inference output:", results)
}
```

---

## Static Compilation

To compile your application into a completely static binary without dynamic library dependencies:

```bash
CGO_ENABLED=1 go build -ldflags="-linkmode external -extldflags '-static -s -w'" -o myapp .
```

Verify that the binary is completely static:
```bash
$ file myapp
myapp: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, stripped

$ ldd myapp
not a dynamic executable
```

### Minimal `FROM scratch` Dockerfile

Because the binary is 100% statically linked, your final Docker image only needs `FROM scratch`:

```dockerfile
FROM scratch
COPY myapp /myapp
ENTRYPOINT ["/myapp"]
```

---

## Project Structure

```text
go-tflite/
├── include/                 # Stripped-down C API headers (~150 KB)
│   └── tensorflow/lite/
│       ├── c/
│       └── core/c/
├── lib/
│   └── linux_amd64/
│       └── libtensorflowlite_c_all.a # Consolidated static library (~14 MB)
├── cgo.go                   # CGO directive flags using ${SRCDIR}
├── model.go                 # Model loading and lifetime management
├── interpreter.go           # Interpreter execution and tensor access
├── tensor.go                # Tensor inspection and typed buffer accessors
├── types.go                 # TensorType enum definitions
├── unsupported.go           # Compile-time guard for unsupported platforms
├── testdata/
│   └── add.bin              # Test TFLite model
├── examples/
│   └── static_app/          # End-to-end static consumer application
├── build.sh                 # Task dispatcher script
└── test.sh                  # Comprehensive verification test script
```

---

## Testing & Verification

Run the test suite on Linux amd64:
```bash
./test.sh
```

Or use the build dispatcher:
```bash
./build.sh test
```

This will:
1. Run `go test -v .` across the library.
2. Statically compile `examples/static_app`.
3. Check the binary with `file` and `ldd` to verify static linking.
4. Run inference locally on the host.
5. Build and execute inside a zero-dependency `FROM scratch` Docker container.

---

## License

Apache License 2.0 (matches TensorFlow).
