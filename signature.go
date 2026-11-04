//go:build linux && amd64

package tflite

/*
#cgo linux,amd64 CFLAGS: -I${SRCDIR}/include
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64 -ltensorflowlite_c_all -lstdc++ -lm -lpthread -ldl
#include <tensorflow/lite/c/c_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// SignatureRunner is used to run inference on a signature.
type SignatureRunner struct {
	cRunner *C.TfLiteSignatureRunner
}

// GetSignatureCount returns the number of signatures.
func (i *Interpreter) GetSignatureCount() int {
	if i == nil || i.cInterp == nil {
		return 0
	}
	return int(C.TfLiteInterpreterGetSignatureCount(i.cInterp))
}

// GetSignatureKey returns the signature key at the given index.
func (i *Interpreter) GetSignatureKey(index int) string {
	if i == nil || i.cInterp == nil {
		return ""
	}
	cStr := C.TfLiteInterpreterGetSignatureKey(i.cInterp, C.int32_t(index))
	if cStr == nil {
		return ""
	}
	return C.GoString(cStr)
}

// GetSignatureRunner returns a new SignatureRunner for the given signature key.
func (i *Interpreter) GetSignatureRunner(key string) (*SignatureRunner, error) {
	if i == nil || i.cInterp == nil {
		return nil, errors.New("interpreter is not initialized")
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	runner := C.TfLiteInterpreterGetSignatureRunner(i.cInterp, cKey)
	if runner == nil {
		return nil, errors.New("signature runner not found")
	}
	return &SignatureRunner{cRunner: runner}, nil
}

// GetInputCount returns the number of input tensors for this signature.
func (s *SignatureRunner) GetInputCount() int {
	if s == nil || s.cRunner == nil {
		return 0
	}
	return int(C.TfLiteSignatureRunnerGetInputCount(s.cRunner))
}

// GetInputName returns the input name at the given index.
func (s *SignatureRunner) GetInputName(index int) string {
	if s == nil || s.cRunner == nil {
		return ""
	}
	cStr := C.TfLiteSignatureRunnerGetInputName(s.cRunner, C.int32_t(index))
	if cStr == nil {
		return ""
	}
	return C.GoString(cStr)
}

// GetInputTensor returns the input tensor for the given name.
func (s *SignatureRunner) GetInputTensor(name string) *Tensor {
	if s == nil || s.cRunner == nil {
		return nil
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	t := C.TfLiteSignatureRunnerGetInputTensor(s.cRunner, cName)
	if t == nil {
		return nil
	}
	return &Tensor{cTensor: (*C.TfLiteTensor)(unsafe.Pointer(t))}
}

// GetOutputCount returns the number of output tensors for this signature.
func (s *SignatureRunner) GetOutputCount() int {
	if s == nil || s.cRunner == nil {
		return 0
	}
	return int(C.TfLiteSignatureRunnerGetOutputCount(s.cRunner))
}

// GetOutputName returns the output name at the given index.
func (s *SignatureRunner) GetOutputName(index int) string {
	if s == nil || s.cRunner == nil {
		return ""
	}
	cStr := C.TfLiteSignatureRunnerGetOutputName(s.cRunner, C.int32_t(index))
	if cStr == nil {
		return ""
	}
	return C.GoString(cStr)
}

// GetOutputTensor returns the output tensor for the given name.
func (s *SignatureRunner) GetOutputTensor(name string) *Tensor {
	if s == nil || s.cRunner == nil {
		return nil
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	t := C.TfLiteSignatureRunnerGetOutputTensor(s.cRunner, cName)
	if t == nil {
		return nil
	}
	return &Tensor{cTensor: (*C.TfLiteTensor)(unsafe.Pointer(t))}
}
