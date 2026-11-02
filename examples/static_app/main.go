package main

import (
	_ "embed"
	"fmt"
	"os"

	tflite "github.com/infinity6-ai/go-tflite"
)

//go:embed add.bin
var defaultModelData []byte

func main() {
	fmt.Printf("=== go-tflite Static Consumer App ===\n")
	fmt.Printf("TensorFlow Lite Version: %s\n", tflite.Version())

	var model *tflite.Model
	var err error

	if len(os.Args) > 1 {
		path := os.Args[1]
		fmt.Printf("Loading external model: %s\n", path)
		model, err = tflite.NewModelFromFile(path)
	} else {
		fmt.Printf("Loading embedded model (size: %d bytes)\n", len(defaultModelData))
		model, err = tflite.NewModelFromBuffer(defaultModelData)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Delete()

	interp, err := tflite.NewInterpreter(model, tflite.Options{NumThreads: 2})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating interpreter: %v\n", err)
		os.Exit(1)
	}
	defer interp.Delete()

	if err := interp.AllocateTensors(); err != nil {
		fmt.Fprintf(os.Stderr, "Error allocating tensors: %v\n", err)
		os.Exit(1)
	}

	inCount := interp.InputTensorCount()
	outCount := interp.OutputTensorCount()
	fmt.Printf("Model tensors: %d input(s), %d output(s)\n", inCount, outCount)

	for i := 0; i < inCount; i++ {
		tensor := interp.GetInputTensor(i)
		fmt.Printf("  Input[%d]: name=%s, type=%s, shape=%v, byteSize=%d\n",
			i, tensor.Name(), tensor.Type(), tensor.Shape(), tensor.ByteSize())

		elemCount := tensor.ByteSize() / 4
		inVals := make([]float32, elemCount)
		for j := range inVals {
			inVals[j] = float32(i+1) * 2.0
		}
		if err := tensor.CopyFromFloat32(inVals); err != nil {
			fmt.Fprintf(os.Stderr, "Error copying input data: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Println("Running inference...")
	if err := interp.Invoke(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running inference: %v\n", err)
		os.Exit(1)
	}

	for i := 0; i < outCount; i++ {
		tensor := interp.GetOutputTensor(i)
		vals, err := tensor.Float32s()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading output tensor: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("  Output[%d]: name=%s, elements=%d, sample values: %v\n",
			i, tensor.Name(), len(vals), vals[:min(len(vals), 8)])
	}

	fmt.Println("=== Inference finished successfully! ===")
}
