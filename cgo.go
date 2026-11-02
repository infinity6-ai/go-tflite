//go:build linux && amd64

package tflite

/*
#cgo linux,amd64 CFLAGS: -I${SRCDIR}/include
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64 -ltensorflowlite_c_all -lstdc++ -lm -lpthread -ldl
#include <tensorflow/lite/c/c_api.h>
#include <stdlib.h>
*/
import "C"

// Version returns the underlying TensorFlow Lite C API version string.
func Version() string {
	return C.GoString(C.TfLiteVersion())
}
