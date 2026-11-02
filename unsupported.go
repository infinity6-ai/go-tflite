//go:build !(linux && amd64)

package tflite

func init() {
	panic("go-tflite: this package currently supports Linux amd64 only")
}
