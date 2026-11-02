//go:build linux && amd64

package tflite

/*
#cgo linux,amd64 CFLAGS: -I${SRCDIR}/include
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64 -ltensorflowlite_c_all -lstdc++ -lm -lpthread -ldl
#include <tensorflow/lite/c/c_api.h>
*/
import "C"
import (
	"errors"
	"fmt"
)

// Options configures the interpreter execution settings.
type Options struct {
	NumThreads int
}

// Interpreter executes inference for a TensorFlow Lite model.
type Interpreter struct {
	cInterp *C.TfLiteInterpreter
}

// NewInterpreter creates a new Interpreter for the given model.
func NewInterpreter(model *Model, opts ...Options) (*Interpreter, error) {
	if model == nil || model.cModel == nil {
		return nil, errors.New("cannot create interpreter from nil model")
	}

	var cOpts *C.TfLiteInterpreterOptions
	if len(opts) > 0 {
		cOpts = C.TfLiteInterpreterOptionsCreate()
		if opts[0].NumThreads > 0 {
			C.TfLiteInterpreterOptionsSetNumThreads(cOpts, C.int32_t(opts[0].NumThreads))
		}
		defer C.TfLiteInterpreterOptionsDelete(cOpts)
	}

	interp := C.TfLiteInterpreterCreate(model.cModel, cOpts)
	if interp == nil {
		return nil, errors.New("failed to create tflite interpreter")
	}

	return &Interpreter{cInterp: interp}, nil
}

// AllocateTensors updates allocations for all tensors.
func (i *Interpreter) AllocateTensors() error {
	if i == nil || i.cInterp == nil {
		return errors.New("interpreter is not initialized")
	}
	status := C.TfLiteInterpreterAllocateTensors(i.cInterp)
	if status != C.kTfLiteOk {
		return fmt.Errorf("failed to allocate tensors, status code: %d", int(status))
	}
	return nil
}

// Invoke runs model inference.
func (i *Interpreter) Invoke() error {
	if i == nil || i.cInterp == nil {
		return errors.New("interpreter is not initialized")
	}
	status := C.TfLiteInterpreterInvoke(i.cInterp)
	if status != C.kTfLiteOk {
		return fmt.Errorf("failed to invoke interpreter, status code: %d", int(status))
	}
	return nil
}

// InputTensorCount returns the number of input tensors.
func (i *Interpreter) InputTensorCount() int {
	if i == nil || i.cInterp == nil {
		return 0
	}
	return int(C.TfLiteInterpreterGetInputTensorCount(i.cInterp))
}

// OutputTensorCount returns the number of output tensors.
func (i *Interpreter) OutputTensorCount() int {
	if i == nil || i.cInterp == nil {
		return 0
	}
	return int(C.TfLiteInterpreterGetOutputTensorCount(i.cInterp))
}

// GetInputTensor returns the input tensor at the given index.
func (i *Interpreter) GetInputTensor(index int) *Tensor {
	if i == nil || i.cInterp == nil {
		return nil
	}
	t := C.TfLiteInterpreterGetInputTensor(i.cInterp, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &Tensor{cTensor: t}
}

// GetOutputTensor returns the output tensor at the given index.
func (i *Interpreter) GetOutputTensor(index int) *Tensor {
	if i == nil || i.cInterp == nil {
		return nil
	}
	t := C.TfLiteInterpreterGetOutputTensor(i.cInterp, C.int32_t(index))
	if t == nil {
		return nil
	}
	return &Tensor{cTensor: t}
}

// Delete frees the interpreter resources.
func (i *Interpreter) Delete() {
	if i != nil && i.cInterp != nil {
		C.TfLiteInterpreterDelete(i.cInterp)
		i.cInterp = nil
	}
}
