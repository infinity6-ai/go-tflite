//go:build linux && amd64

package tflite_test

import (
	"os"
	"testing"

	tflite "github.com/infinity6-ai/go-tflite"
)

func TestUnitVersion(t *testing.T) {
	v := tflite.Version()
	if v == "" {
		t.Fatal("expected non-empty version string")
	}
	t.Logf("TFLite Version: %s", v)
}

func TestUnitModelFromFile(t *testing.T) {
	model, err := tflite.NewModelFromFile("testdata/add.bin")
	if err != nil {
		t.Fatalf("failed to load model from file: %v", err)
	}
	defer model.Delete()
}

func TestUnitInference(t *testing.T) {
	data, err := os.ReadFile("testdata/add.bin")
	if err != nil {
		t.Fatalf("failed to read testdata/add.bin: %v", err)
	}

	model, err := tflite.NewModelFromBuffer(data)
	if err != nil {
		t.Fatalf("failed to load model from buffer: %v", err)
	}
	defer model.Delete()

	interp, err := tflite.NewInterpreter(model, tflite.Options{NumThreads: 2})
	if err != nil {
		t.Fatalf("failed to create interpreter: %v", err)
	}
	defer interp.Delete()

	if err := interp.AllocateTensors(); err != nil {
		t.Fatalf("failed to allocate tensors: %v", err)
	}

	if interp.InputTensorCount() != 1 {
		t.Fatalf("expected 1 input tensor, got %d", interp.InputTensorCount())
	}
	if interp.OutputTensorCount() != 1 {
		t.Fatalf("expected 1 output tensor, got %d", interp.OutputTensorCount())
	}

	inTensor := interp.GetInputTensor(0)
	if inTensor == nil {
		t.Fatal("input tensor is nil")
	}
	t.Logf("Input tensor: name=%s, type=%s, shape=%v, byteSize=%d",
		inTensor.Name(), inTensor.Type(), inTensor.Shape(), inTensor.ByteSize())

	elemCount := inTensor.ByteSize() / 4
	inValues := make([]float32, elemCount)
	for i := range inValues {
		inValues[i] = 2.0
	}

	if err := inTensor.CopyFromFloat32(inValues); err != nil {
		t.Fatalf("failed to copy input float32s: %v", err)
	}

	if err := interp.Invoke(); err != nil {
		t.Fatalf("failed to invoke interpreter: %v", err)
	}

	outTensor := interp.GetOutputTensor(0)
	if outTensor == nil {
		t.Fatal("output tensor is nil")
	}

	outValues, err := outTensor.Float32s()
	if err != nil {
		t.Fatalf("failed to read output float32s: %v", err)
	}

	t.Logf("Output tensor: name=%s, elements=%d, first 4 values=%v",
		outTensor.Name(), len(outValues), outValues[:min(len(outValues), 4)])

	if len(outValues) == 0 {
		t.Fatal("expected non-empty output values")
	}
	// The add.bin model adds input to an internal constant (4.0), so 2.0 + 4.0 = 6.0
	if outValues[0] != 6.0 {
		t.Fatalf("expected output value 6.0, got %f", outValues[0])
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
