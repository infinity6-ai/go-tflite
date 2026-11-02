//go:build linux && amd64

package tflite

import "fmt"

// TensorType represents data type of a tensor.
type TensorType int

const (
	NoType     TensorType = 0
	Float32    TensorType = 1
	Int32      TensorType = 2
	UInt8      TensorType = 3
	Int64      TensorType = 4
	String     TensorType = 5
	Bool       TensorType = 6
	Int16      TensorType = 7
	Complex64  TensorType = 8
	Int8       TensorType = 9
	Float16    TensorType = 10
	Float64    TensorType = 11
	Complex128 TensorType = 12
	UInt64     TensorType = 13
	Resource   TensorType = 14
	Variant    TensorType = 15
	UInt32     TensorType = 16
	UInt16     TensorType = 17
	Int4       TensorType = 18
)

func (t TensorType) String() string {
	switch t {
	case Float32:
		return "Float32"
	case Int32:
		return "Int32"
	case UInt8:
		return "UInt8"
	case Int64:
		return "Int64"
	case String:
		return "String"
	case Bool:
		return "Bool"
	case Int16:
		return "Int16"
	case Complex64:
		return "Complex64"
	case Int8:
		return "Int8"
	case Float16:
		return "Float16"
	case Float64:
		return "Float64"
	case Complex128:
		return "Complex128"
	case UInt64:
		return "UInt64"
	case Resource:
		return "Resource"
	case Variant:
		return "Variant"
	case UInt32:
		return "UInt32"
	case UInt16:
		return "UInt16"
	case Int4:
		return "Int4"
	default:
		return fmt.Sprintf("Unknown(%d)", int(t))
	}
}
