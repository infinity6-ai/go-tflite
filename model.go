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
	"fmt"
	"unsafe"
)

// Model represents a loaded TensorFlow Lite model.
type Model struct {
	cModel *C.TfLiteModel
}

// NewModelFromFile loads a TensorFlow Lite model from a file path.
func NewModelFromFile(path string) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	m := C.TfLiteModelCreateFromFile(cPath)
	if m == nil {
		return nil, fmt.Errorf("failed to load tflite model from file: %s", path)
	}
	return &Model{cModel: m}, nil
}

// NewModelFromBuffer loads a TensorFlow Lite model from an in-memory byte slice.
func NewModelFromBuffer(data []byte) (*Model, error) {
	if len(data) == 0 {
		return nil, errors.New("cannot create model from empty buffer")
	}

	m := C.TfLiteModelCreate(unsafe.Pointer(&data[0]), C.size_t(len(data)))
	if m == nil {
		return nil, errors.New("failed to load tflite model from buffer")
	}
	return &Model{cModel: m}, nil
}

// Delete frees the resources associated with the Model.
func (m *Model) Delete() {
	if m != nil && m.cModel != nil {
		C.TfLiteModelDelete(m.cModel)
		m.cModel = nil
	}
}
