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
	"unsafe"
)

// Tensor represents an input or output tensor of an interpreter.
type Tensor struct {
	cTensor *C.TfLiteTensor
}

// Name returns the name of the tensor.
func (t *Tensor) Name() string {
	if t == nil || t.cTensor == nil {
		return ""
	}
	return C.GoString(C.TfLiteTensorName(t.cTensor))
}

// Type returns the data type of the tensor.
func (t *Tensor) Type() TensorType {
	if t == nil || t.cTensor == nil {
		return NoType
	}
	return TensorType(C.TfLiteTensorType(t.cTensor))
}

// NumDims returns the number of dimensions of the tensor.
func (t *Tensor) NumDims() int {
	if t == nil || t.cTensor == nil {
		return 0
	}
	return int(C.TfLiteTensorNumDims(t.cTensor))
}

// Dim returns the size of the dimension at the given index.
func (t *Tensor) Dim(index int) int {
	if t == nil || t.cTensor == nil {
		return 0
	}
	return int(C.TfLiteTensorDim(t.cTensor, C.int32_t(index)))
}

// Shape returns the full dimension shape of the tensor.
func (t *Tensor) Shape() []int {
	n := t.NumDims()
	if n == 0 {
		return nil
	}
	shape := make([]int, n)
	for i := 0; i < n; i++ {
		shape[i] = t.Dim(i)
	}
	return shape
}

// ByteSize returns the size of the underlying tensor data buffer in bytes.
func (t *Tensor) ByteSize() int {
	if t == nil || t.cTensor == nil {
		return 0
	}
	return int(C.TfLiteTensorByteSize(t.cTensor))
}

// CopyFromBuffer copies raw bytes into the tensor.
func (t *Tensor) CopyFromBuffer(data []byte) error {
	if t == nil || t.cTensor == nil {
		return errors.New("tensor is nil")
	}
	if len(data) == 0 {
		return nil
	}
	status := C.TfLiteTensorCopyFromBuffer(t.cTensor, unsafe.Pointer(&data[0]), C.size_t(len(data)))
	if status != C.kTfLiteOk {
		return fmt.Errorf("failed to copy data to tensor, status code: %d", int(status))
	}
	return nil
}

// CopyToBuffer copies raw bytes from the tensor into the provided slice.
func (t *Tensor) CopyToBuffer(dst []byte) error {
	if t == nil || t.cTensor == nil {
		return errors.New("tensor is nil")
	}
	if len(dst) == 0 {
		return nil
	}
	status := C.TfLiteTensorCopyToBuffer(t.cTensor, unsafe.Pointer(&dst[0]), C.size_t(len(dst)))
	if status != C.kTfLiteOk {
		return fmt.Errorf("failed to copy data from tensor, status code: %d", int(status))
	}
	return nil
}

// CopyFromFloat32 copies a slice of float32 values into the tensor.
func (t *Tensor) CopyFromFloat32(data []float32) error {
	if len(data) == 0 {
		return nil
	}
	byteLen := len(data) * 4
	if byteLen > t.ByteSize() {
		return fmt.Errorf("data size (%d bytes) exceeds tensor size (%d bytes)", byteLen, t.ByteSize())
	}
	status := C.TfLiteTensorCopyFromBuffer(t.cTensor, unsafe.Pointer(&data[0]), C.size_t(byteLen))
	if status != C.kTfLiteOk {
		return fmt.Errorf("failed to copy float32 data to tensor, status code: %d", int(status))
	}
	return nil
}

// Float32s reads the tensor buffer as a slice of float32 values.
func (t *Tensor) Float32s() ([]float32, error) {
	bytes := t.ByteSize()
	if bytes%4 != 0 {
		return nil, fmt.Errorf("tensor byte size (%d) is not a multiple of 4", bytes)
	}
	count := bytes / 4
	out := make([]float32, count)
	if count == 0 {
		return out, nil
	}
	status := C.TfLiteTensorCopyToBuffer(t.cTensor, unsafe.Pointer(&out[0]), C.size_t(bytes))
	if status != C.kTfLiteOk {
		return nil, fmt.Errorf("failed to copy float32 data from tensor, status code: %d", int(status))
	}
	return out, nil
}

// CopyFromInt32 copies a slice of int32 values into the tensor.
func (t *Tensor) CopyFromInt32(data []int32) error {
	if len(data) == 0 {
		return nil
	}
	byteLen := len(data) * 4
	if byteLen > t.ByteSize() {
		return fmt.Errorf("data size (%d bytes) exceeds tensor size (%d bytes)", byteLen, t.ByteSize())
	}
	status := C.TfLiteTensorCopyFromBuffer(t.cTensor, unsafe.Pointer(&data[0]), C.size_t(byteLen))
	if status != C.kTfLiteOk {
		return fmt.Errorf("failed to copy int32 data to tensor, status code: %d", int(status))
	}
	return nil
}

// Int32s reads the tensor buffer as a slice of int32 values.
func (t *Tensor) Int32s() ([]int32, error) {
	bytes := t.ByteSize()
	if bytes%4 != 0 {
		return nil, fmt.Errorf("tensor byte size (%d) is not a multiple of 4", bytes)
	}
	count := bytes / 4
	out := make([]int32, count)
	if count == 0 {
		return out, nil
	}
	status := C.TfLiteTensorCopyToBuffer(t.cTensor, unsafe.Pointer(&out[0]), C.size_t(bytes))
	if status != C.kTfLiteOk {
		return nil, fmt.Errorf("failed to copy int32 data from tensor, status code: %d", int(status))
	}
	return out, nil
}

// CopyFromUint8 copies a slice of uint8 values into the tensor.
func (t *Tensor) CopyFromUint8(data []uint8) error {
	return t.CopyFromBuffer(data)
}

// Uint8s reads the tensor buffer as a slice of uint8 values.
func (t *Tensor) Uint8s() ([]uint8, error) {
	bytes := t.ByteSize()
	out := make([]uint8, bytes)
	if err := t.CopyToBuffer(out); err != nil {
		return nil, err
	}
	return out, nil
}
