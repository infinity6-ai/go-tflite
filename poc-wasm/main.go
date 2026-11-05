package main

import (
	"context"
	_ "embed"
	"encoding/binary"
	"fmt"
	"log"
	"math"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/emscripten"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

//go:embed tflite.wasm
var tfliteWasm []byte

//go:embed add.bin
var modelData []byte

func main() {
	ctx := context.Background()

	// 1. Create a pure-Go WebAssembly runtime
	r := wazero.NewRuntime(ctx)
	defer r.Close(ctx)

	// 2. Instantiate WASI (preview 1)
	wasi_snapshot_preview1.MustInstantiate(ctx, r)

	// 3. Compile the WASM binary
	compiledModule, err := r.CompileModule(ctx, tfliteWasm)
	if err != nil {
		log.Fatalf("failed to compile wasm module: %v", err)
	}

	// 4. Instantiate Emscripten host imports
	_, err = emscripten.InstantiateForModule(ctx, r, compiledModule)
	if err != nil {
		log.Fatalf("failed to instantiate emscripten imports: %v", err)
	}

	// 5. Instantiate the module
	mod, err := r.InstantiateModule(ctx, compiledModule, wazero.NewModuleConfig().WithName("tflite"))
	if err != nil {
		log.Fatalf("failed to instantiate wasm module: %v", err)
	}

	// Helper to lookup exported function
	fn := func(name string) api.Function {
		f := mod.ExportedFunction(name)
		if f == nil {
			f = mod.ExportedFunction("_" + name)
		}
		if f == nil {
			log.Fatalf("Exported function %q not found", name)
		}
		return f
	}

	mallocFn := fn("malloc")
	freeFn := fn("free")
	versionFn := fn("TfLiteVersion")
	modelCreateFn := fn("TfLiteModelCreate")
	modelDeleteFn := fn("TfLiteModelDelete")
	interpCreateFn := fn("TfLiteInterpreterCreate")
	interpDeleteFn := fn("TfLiteInterpreterDelete")
	allocTensorsFn := fn("TfLiteInterpreterAllocateTensors")
	invokeFn := fn("TfLiteInterpreterInvoke")
	getInputTensorFn := fn("TfLiteInterpreterGetInputTensor")
	getOutputTensorFn := fn("TfLiteInterpreterGetOutputTensor")
	tensorByteSizeFn := fn("TfLiteTensorByteSize")
	copyFromBufferFn := fn("TfLiteTensorCopyFromBuffer")
	copyToBufferFn := fn("TfLiteTensorCopyToBuffer")

	// 6. Test TfLiteVersion()
	res, err := versionFn.Call(ctx)
	if err != nil {
		log.Fatalf("failed to call TfLiteVersion: %v", err)
	}
	mem := mod.Memory()
	var versionStr []byte
	for i := uint32(res[0]); ; i++ {
		b, ok := mem.ReadByte(i)
		if !ok || b == 0 {
			break
		}
		versionStr = append(versionStr, b)
	}
	fmt.Printf("[1/5] TensorFlow Lite Version (WASM / pure Go): %s\n", string(versionStr))

	// 7. Load model into WASM linear memory
	modelSize := uint64(len(modelData))
	res, err = mallocFn.Call(ctx, modelSize)
	if err != nil {
		log.Fatalf("malloc failed: %v", err)
	}
	wasmModelPtr := uint32(res[0])
	defer freeFn.Call(ctx, uint64(wasmModelPtr))

	if !mem.Write(wasmModelPtr, modelData) {
		log.Fatalf("failed to write model to wasm memory")
	}

	res, err = modelCreateFn.Call(ctx, uint64(wasmModelPtr), modelSize)
	if err != nil || res[0] == 0 {
		log.Fatalf("TfLiteModelCreate failed")
	}
	modelPtr := res[0]
	defer modelDeleteFn.Call(ctx, modelPtr)
	fmt.Printf("[2/5] Model loaded successfully (%d bytes)\n", modelSize)

	// 8. Create interpreter & allocate tensors
	res, err = interpCreateFn.Call(ctx, modelPtr, 0)
	if err != nil || res[0] == 0 {
		log.Fatalf("TfLiteInterpreterCreate failed")
	}
	interpPtr := res[0]
	defer interpDeleteFn.Call(ctx, interpPtr)

	res, err = allocTensorsFn.Call(ctx, interpPtr)
	if err != nil || res[0] != 0 {
		log.Fatalf("TfLiteInterpreterAllocateTensors failed (code %d)", res[0])
	}
	fmt.Printf("[3/5] Interpreter created and tensors allocated\n")

	// 9. Prepare Input Tensor (input = 2.0f)
	res, err = getInputTensorFn.Call(ctx, interpPtr, 0)
	if err != nil || res[0] == 0 {
		log.Fatalf("GetInputTensor failed")
	}
	inTensorPtr := res[0]

	res, err = tensorByteSizeFn.Call(ctx, inTensorPtr)
	if err != nil {
		log.Fatalf("tensorByteSize failed: %v", err)
	}
	expectedByteSize := res[0]
	fmt.Printf("  Input tensor byte size: %d bytes\n", expectedByteSize)

	res, err = mallocFn.Call(ctx, expectedByteSize)
	if err != nil {
		log.Fatalf("malloc input buffer failed")
	}
	inBufPtr := uint32(res[0])
	defer freeFn.Call(ctx, uint64(inBufPtr))

	inputData := make([]float32, expectedByteSize/4)
	for i := range inputData {
		inputData[i] = 2.0
	}
	inputBytes := make([]byte, expectedByteSize)
	for i, v := range inputData {
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(v))
	}
	mem.Write(inBufPtr, inputBytes)

	res, err = copyFromBufferFn.Call(ctx, inTensorPtr, uint64(inBufPtr), expectedByteSize)
	if err != nil || res[0] != 0 {
		log.Fatalf("CopyFromBuffer failed (status=%d, err=%v)", res[0], err)
	}
	fmt.Printf("[4/5] Set input tensor to: 2.0\n")

	// 10. Invoke inference
	res, err = invokeFn.Call(ctx, interpPtr)
	if err != nil || res[0] != 0 {
		log.Fatalf("TfLiteInterpreterInvoke failed (code %d)", res[0])
	}

	// 11. Read Output Tensor (expected: 2.0 + 4.0 = 6.0)
	res, err = getOutputTensorFn.Call(ctx, interpPtr, 0)
	if err != nil || res[0] == 0 {
		log.Fatalf("GetOutputTensor failed")
	}
	outTensorPtr := res[0]

	res, err = tensorByteSizeFn.Call(ctx, outTensorPtr)
	if err != nil {
		log.Fatalf("tensorByteSize failed on output: %v", err)
	}
	outByteSize := res[0]

	res, err = mallocFn.Call(ctx, outByteSize)
	if err != nil {
		log.Fatalf("malloc output buffer failed")
	}
	outBufPtr := uint32(res[0])
	defer freeFn.Call(ctx, uint64(outBufPtr))

	res, err = copyToBufferFn.Call(ctx, outTensorPtr, uint64(outBufPtr), outByteSize)
	if err != nil || res[0] != 0 {
		log.Fatalf("CopyToBuffer failed (code=%d)", res[0])
	}

	outBytes, ok := mem.Read(outBufPtr, uint32(outByteSize))
	if !ok {
		log.Fatalf("failed to read output from wasm memory")
	}
	result := math.Float32frombits(binary.LittleEndian.Uint32(outBytes[:4]))

	fmt.Printf("[5/5] Inference result (first element): %f (Expected: 6.000000)\n", result)
	if result == 6.0 {
		fmt.Printf("🎉 FULL POC VERIFIED: WebAssembly TFLite inference works 100%% pure Go (CGO_ENABLED=0)!\n")
	} else {
		log.Fatalf("Unexpected inference result: %f", result)
	}
}
