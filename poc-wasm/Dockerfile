# syntax=docker/dockerfile:1.4
# Build TensorFlow Lite WebAssembly (WASM) Module using Emscripten
FROM emscripten/emsdk:3.1.50

# Install build dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    file \
    && rm -rf /var/lib/apt/lists/*

# Clone TensorFlow (shallow clone of a specific stable version)
WORKDIR /tensorflow_src
RUN git clone --depth 1 --branch v2.16.2 https://github.com/tensorflow/tensorflow.git .

# Configure and build the C API static library using emcmake
WORKDIR /build
RUN emcmake cmake /tensorflow_src/tensorflow/lite/c \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_C_BUILD_SHARED_LIBS=OFF \
    -DTFLITE_ENABLE_XNNPACK=OFF \
    -DTFLITE_ENABLE_GPU=OFF

RUN emmake cmake --build . -j $(nproc) -t tensorflowlite_c

# Provide ClassicLocale definition required by FlatBuffers / TFLite in WebAssembly
COPY flatbuffers_locale_fix.cc /build/flatbuffers_locale_fix.cc
RUN em++ -c /build/flatbuffers_locale_fix.cc -o /build/flatbuffers_locale_fix.o -O3

# Combine all generated static libraries into a single monolithic archive
# Note: we use Emscripten's 'emar' which handles LLVM bitcode / wasm objects
RUN echo "CREATE libtensorflowlite_c_all.a" > combine.mri && \
    find . -name "*.a" ! -name "libtensorflowlite_c_all.a" | while read -r f; do \
        echo "ADDLIB $f" >> combine.mri; \
    done && \
    echo "SAVE" >> combine.mri && \
    echo "END" >> combine.mri && \
    emar -M < combine.mri && \
    emranlib libtensorflowlite_c_all.a

# Link into a standalone WebAssembly module
# -s STANDALONE_WASM=1: Target WASI without requiring JS runtime
# -s ALLOW_MEMORY_GROWTH=1: Allow linear memory to expand as TFLite allocates
# --no-entry: It's a library, so we do not expect a `main` function
# -s EXPORTED_FUNCTIONS: Only export the C API functions that our Go wrapper will need
RUN emcc flatbuffers_locale_fix.o libtensorflowlite_c_all.a -o tflite.wasm -O3 \
    -s STANDALONE_WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
    --no-entry \
    -s EXPORTED_FUNCTIONS="['_TfLiteInterpreterAllocateTensors', '_TfLiteInterpreterCreate', '_TfLiteInterpreterDelete', '_TfLiteInterpreterGetInputTensor', '_TfLiteInterpreterGetInputTensorCount', '_TfLiteInterpreterGetOutputTensor', '_TfLiteInterpreterGetOutputTensorCount', '_TfLiteInterpreterGetSignatureCount', '_TfLiteInterpreterGetSignatureKey', '_TfLiteInterpreterGetSignatureRunner', '_TfLiteInterpreterInvoke', '_TfLiteInterpreterOptionsCreate', '_TfLiteInterpreterOptionsDelete', '_TfLiteInterpreterOptionsSetNumThreads', '_TfLiteModelCreate', '_TfLiteModelCreateFromFile', '_TfLiteModelDelete', '_TfLiteSignatureRunnerGetInputCount', '_TfLiteSignatureRunnerGetInputName', '_TfLiteSignatureRunnerGetInputTensor', '_TfLiteSignatureRunnerGetOutputCount', '_TfLiteSignatureRunnerGetOutputName', '_TfLiteSignatureRunnerGetOutputTensor', '_TfLiteTensorByteSize', '_TfLiteTensorCopyFromBuffer', '_TfLiteTensorCopyToBuffer', '_TfLiteTensorDim', '_TfLiteTensorName', '_TfLiteTensorNumDims', '_TfLiteTensorType', '_TfLiteVersion', '_malloc', '_free']"

# Verify that the output is indeed a WebAssembly binary
RUN file tflite.wasm

# The final image contains just the WASM binary so it can be easily extracted
FROM scratch
COPY --from=0 /build/tflite.wasm /tflite.wasm
