//go:build linux && amd64

package tflite

import "embed"

//go:embed lib/linux_amd64/libtensorflowlite_c_all.a
var _ []byte

//go:embed include
var _ embed.FS
