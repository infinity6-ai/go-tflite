#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="target"
BINARY_PATH="${TARGET_DIR}/go-tflite-app"

echo "=== 1. Running go-tflite package tests ==="
go test -v .

echo ""
echo "=== 2. Building consumer static application ==="
mkdir -p "${TARGET_DIR}"
(
  cd examples/static_app
  CGO_ENABLED=1 go build -ldflags="-linkmode external -extldflags '-static -s -w'" -o "../../${BINARY_PATH}" .
)

echo ""
echo "=== 3. Inspecting compiled binary ==="
file "${BINARY_PATH}"
if command -v ldd >/dev/null 2>&1; then
    ldd "${BINARY_PATH}" || true
fi

echo ""
echo "=== 4. Running static binary locally on host ==="
"./${BINARY_PATH}"

echo ""
echo "=== 5. Testing inside minimal scratch Docker container ==="
docker build -t go-tflite:test -f - . << 'EOF'
FROM scratch
COPY target/go-tflite-app /app
ENTRYPOINT ["/app"]
EOF
docker run --rm go-tflite:test

echo ""
echo "=== All tests passed successfully! ==="
