# syntax=docker/dockerfile:1.4
# Stage 1: Build TensorFlow Lite Static Library from source
FROM debian:bookworm-slim AS tflite-builder

# Install build dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Clone TensorFlow (shallow clone of a specific stable version)
WORKDIR /tensorflow_src
RUN git clone --depth 1 --branch v2.16.2 https://github.com/tensorflow/tensorflow.git .

# Configure and build the C API static library
WORKDIR /build
RUN cmake /tensorflow_src/tensorflow/lite/c \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_C_BUILD_SHARED_LIBS=OFF \
    -DTFLITE_ENABLE_XNNPACK=OFF \
    -DTFLITE_ENABLE_GPU=OFF

RUN cmake --build . -j $(nproc) -t tensorflowlite_c

# Combine all generated static libraries into a single monolithic archive
RUN echo "CREATE libtensorflowlite_c_all.a" > combine.mri && \
    find . -name "*.a" ! -name "libtensorflowlite_c_all.a" | while read -r f; do \
        echo "ADDLIB $f" >> combine.mri; \
    done && \
    echo "SAVE" >> combine.mri && \
    echo "END" >> combine.mri && \
    ar -M < combine.mri && \
    ranlib libtensorflowlite_c_all.a


# Stage 2: Build the static Go application
FROM golang:1.26.1-bookworm AS go-builder

# Install C/C++ static compilation dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the TFLite source (for headers) and the monolithic static library
COPY --from=tflite-builder /tensorflow_src /tensorflow
COPY --from=tflite-builder /build/libtensorflowlite_c_all.a /build/libtensorflowlite_c_all.a
# Copy the test model so it can be embedded via //go:embed
COPY --from=tflite-builder /tensorflow_src/tensorflow/lite/testdata/add.bin /app/add.bin

# Copy Go source code from local poc directory
COPY go.mod ./
COPY main.go ./

# Build statically linked Go binary
ENV CGO_ENABLED=1
RUN go build -ldflags="-linkmode external -extldflags '-static -s -w'" -o /bin/tflite-app .

# Verify the binary is actually statically linked
RUN file /bin/tflite-app | grep "statically linked" || (echo "Binary is not statically linked!" && exit 1)


# Stage 3: Scratch Runtime
# The final image contains ONLY the static binary (0 dependencies, 0 OS files)
FROM scratch

COPY --from=go-builder /bin/tflite-app /tflite-app

ENTRYPOINT ["/tflite-app"]
