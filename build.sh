#!/bin/bash -xe

function cmd_clean() {
  local _k=""
  find -name target | while read _k; do
    rm -rf "$_k"
  done
  rm -f examples/static_app/static_app examples/static_app/static-app
}

function cmd_build() {
  go build ./...
  mkdir -p target
  (cd examples/static_app && CGO_ENABLED=1 go build -ldflags="-linkmode external -extldflags '-static -s -w'" -o ../../target/go-tflite-app .)
}

function cmd_go() {
  go "$@"
}

function cmd_update() {
  go mod tidy
  (cd examples/static_app && go mod tidy)
  cmd_codegen
}

function cmd_run() {
  (cd examples/static_app && go run . "$@")
}

function cmd_codegen() {
  go generate ./...
}

function cmd_test_unit() {
  go test -run '^TestUnit.*$' ./... "$@"
}

function cmd_test_remote() {
  go test -run '^TestRemote.*$' ./... "$@"
}

function cmd_test() {
  cmd_test_unit
}

function cmd_release() {
  local _version="${1?"version is required (e.g. v0.0.1)"}"

  if [[ "$_version" != v* ]]; then
    echo "Error: version must start with 'v' (e.g. v0.0.1)" >&2
    exit 1
  fi

  if git rev-parse "$_version" >/dev/null 2>&1; then
    echo "Error: tag '$_version' already exists" >&2
    exit 1
  fi

  if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working directory is not clean. Commit or stash changes first." >&2
    exit 1
  fi

  local _branch
  _branch="$(git branch --show-current)"
  if [ -z "$_branch" ]; then
    echo "Error: cannot release from detached HEAD" >&2
    exit 1
  fi

  cmd_test

  git tag -a "$_version" -m "Release $_version"
  git push origin "$_branch"
  git push origin "$_version"

  echo "Successfully released ${_version} on branch ${_branch}!"
}

_cmd="${1?"cmd is required"}"; shift; 
cd "$(dirname "$0")"; 
"cmd_${_cmd}" "$@"
