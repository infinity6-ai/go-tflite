package main

/*
#cgo CFLAGS: -I/tensorflow
#cgo LDFLAGS: -L/build -ltensorflowlite_c_all -lstdc++ -lm -lpthread -ldl
#include <tensorflow/lite/c/c_api.h>
#include <stdlib.h>
*/
import "C"
import (
	_ "embed"
	"fmt"
	"os"
	"unsafe"
)

//go:embed add.bin
var defaultModelData []byte

func main() {
	v := C.TfLiteVersion()
	fmt.Printf("== TensorFlow Lite Static Go POC ==\n")
	fmt.Printf("TFLite C API Version: %s\n", C.GoString(v))

	var model *C.TfLiteModel
	var modelPath string

	if len(os.Args) > 1 {
		modelPath = os.Args[1]
		fmt.Printf("Loading external model: %s\n", modelPath)
		cPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cPath))
		model = C.TfLiteModelCreateFromFile(cPath)
	} else {
		fmt.Printf("Loading embedded default model (add.bin), size: %d bytes\n", len(defaultModelData))
		model = C.TfLiteModelCreate(unsafe.Pointer(&defaultModelData[0]), C.size_t(len(defaultModelData)))
	}

	if model == nil {
		fmt.Fprintf(os.Stderr, "Failed to load model\n")
		os.Exit(1)
	}
	defer C.TfLiteModelDelete(model)

	options := C.TfLiteInterpreterOptionsCreate()
	C.TfLiteInterpreterOptionsSetNumThreads(options, 2)
	defer C.TfLiteInterpreterOptionsDelete(options)

	interpreter := C.TfLiteInterpreterCreate(model, options)
	if interpreter == nil {
		fmt.Fprintf(os.Stderr, "Failed to create interpreter\n")
		os.Exit(1)
	}
	defer C.TfLiteInterpreterDelete(interpreter)

	if status := C.TfLiteInterpreterAllocateTensors(interpreter); status != C.kTfLiteOk {
		fmt.Fprintf(os.Stderr, "Failed to allocate tensors (status=%d)\n", int(status))
		os.Exit(1)
	}

	inCount := int(C.TfLiteInterpreterGetInputTensorCount(interpreter))
	outCount := int(C.TfLiteInterpreterGetOutputTensorCount(interpreter))
	fmt.Printf("Model allocated successfully: %d input(s), %d output(s)\n", inCount, outCount)

	for i := 0; i < inCount; i++ {
		tensor := C.TfLiteInterpreterGetInputTensor(interpreter, C.int32_t(i))
		name := C.GoString(C.TfLiteTensorName(tensor))
		byteSize := int(C.TfLiteTensorByteSize(tensor))
		numDims := int(C.TfLiteTensorNumDims(tensor))
		fmt.Printf("  Input[%d]: name=%s, bytes=%d, numDims=%d\n", i, name, byteSize, numDims)

		count := byteSize / 4
		if count > 0 {
			inputData := make([]float32, count)
			for j := range inputData {
				inputData[j] = float32(i+1) * 2.0
			}

			status := C.TfLiteTensorCopyFromBuffer(tensor, unsafe.Pointer(&inputData[0]), C.size_t(byteSize))
			if status != C.kTfLiteOk {
				fmt.Fprintf(os.Stderr, "Failed to copy data into input %d\n", i)
				os.Exit(1)
			}
		}
	}

	fmt.Println("Invoking inference...")
	if status := C.TfLiteInterpreterInvoke(interpreter); status != C.kTfLiteOk {
		fmt.Fprintf(os.Stderr, "Failed to invoke interpreter (status=%d)\n", int(status))
		os.Exit(1)
	}

	for i := 0; i < outCount; i++ {
		tensor := C.TfLiteInterpreterGetOutputTensor(interpreter, C.int32_t(i))
		name := C.GoString(C.TfLiteTensorName(tensor))
		byteSize := int(C.TfLiteTensorByteSize(tensor))
		count := byteSize / 4

		if count > 0 {
			outputData := make([]float32, count)
			status := C.TfLiteTensorCopyToBuffer(tensor, unsafe.Pointer(&outputData[0]), C.size_t(byteSize))
			if status != C.kTfLiteOk {
				fmt.Fprintf(os.Stderr, "Failed to copy data from output %d\n", i)
				os.Exit(1)
			}
			fmt.Printf("  Output[%d]: name=%s, elements=%d, sample values: %v\n", i, name, count, outputData[:min(count, 8)])
		}
	}

	fmt.Println("== Inference finished successfully! ==")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
