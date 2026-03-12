#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.24.0}"
ORT_FLAVOR="${ORT_FLAVOR:-cpu}"
case "${ORT_FLAVOR}" in
  cpu)
    ORT_BASENAME="${ORT_BASENAME:-onnxruntime-linux-x64-${ORT_VERSION}}"
    ;;
  gpu)
    ORT_BASENAME="${ORT_BASENAME:-onnxruntime-linux-x64-gpu-${ORT_VERSION}}"
    ;;
  gpu_cuda12)
    ORT_BASENAME="${ORT_BASENAME:-onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}}"
    ;;
  gpu_cuda13)
    ORT_BASENAME="${ORT_BASENAME:-onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}}"
    ;;
  *)
    ORT_BASENAME="${ORT_BASENAME:-onnxruntime-linux-x64-${ORT_FLAVOR}-${ORT_VERSION}}"
    ;;
esac
ORT_ARCHIVE="${ORT_BASENAME}.tgz"
ORT_URL="${ORT_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}}"
INSTALL_ROOT="${1:-.local/onnxruntime}"
DOWNLOAD_DIR="${INSTALL_ROOT}/downloads"
EXTRACT_DIR="${INSTALL_ROOT}/${ORT_BASENAME}"
CURRENT_LINK="${INSTALL_ROOT}/current"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

need_cmd tar
if command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl -fL --retry 3 -o"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget -O"
else
  echo "missing required downloader: curl or wget" >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"

if [[ ! -f "${DOWNLOAD_DIR}/${ORT_ARCHIVE}" ]]; then
  echo "downloading ${ORT_URL}" >&2
  # shellcheck disable=SC2086
  ${DOWNLOADER} "${DOWNLOAD_DIR}/${ORT_ARCHIVE}" "${ORT_URL}"
else
  echo "using cached archive ${DOWNLOAD_DIR}/${ORT_ARCHIVE}" >&2
fi

rm -rf "${EXTRACT_DIR}"
mkdir -p "${INSTALL_ROOT}"
tar -xzf "${DOWNLOAD_DIR}/${ORT_ARCHIVE}" -C "${INSTALL_ROOT}"
if [[ ! -d "${EXTRACT_DIR}" ]]; then
  ALT_EXTRACT_DIR="$(tar -tzf "${DOWNLOAD_DIR}/${ORT_ARCHIVE}" | sed -n '1s#/.*##p')"
  if [[ -n "${ALT_EXTRACT_DIR}" && -d "${INSTALL_ROOT}/${ALT_EXTRACT_DIR}" ]]; then
    EXTRACT_DIR="${INSTALL_ROOT}/${ALT_EXTRACT_DIR}"
  fi
fi

if [[ ! -f "${EXTRACT_DIR}/lib/libonnxruntime.so" ]]; then
  VERSIONED_LIB="$(find "${EXTRACT_DIR}/lib" -maxdepth 1 -type f -name 'libonnxruntime.so.*' | head -n1 || true)"
  if [[ -n "${VERSIONED_LIB}" ]]; then
    ln -sfn "$(basename "${VERSIONED_LIB}")" "${EXTRACT_DIR}/lib/libonnxruntime.so"
  fi
fi

if [[ ! -f "${EXTRACT_DIR}/lib/libonnxruntime.so" ]]; then
  echo "install failed: ${EXTRACT_DIR}/lib/libonnxruntime.so not found" >&2
  exit 1
fi

ln -sfn "$(basename "${EXTRACT_DIR}")" "${CURRENT_LINK}"

cat <<EOF
Installed ONNX Runtime to:
  ${EXTRACT_DIR}
Stable project-local path:
  ${CURRENT_LINK}

Export these before running CUDA binaries:
  export ORT_DYLIB_PATH="$(cd "${EXTRACT_DIR}/lib" && pwd)/libonnxruntime.so"
  export LD_LIBRARY_PATH="$(cd "${EXTRACT_DIR}/lib" && pwd):\${LD_LIBRARY_PATH:-}"

Expected shared libraries:
  ${EXTRACT_DIR}/lib/libonnxruntime.so
  ${EXTRACT_DIR}/lib/libonnxruntime_providers_shared.so (provider packages only)
  ${EXTRACT_DIR}/lib/libonnxruntime_providers_cuda.so (GPU packages only)
EOF
