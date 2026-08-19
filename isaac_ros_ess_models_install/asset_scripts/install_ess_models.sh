#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Download and tao-convert ESS models.
# * Models will be stored in the isaac_ros_assets dir
# * The script must be called with the --eula argument prior to downloading.

set -e

if [ -n "$TENSORRT_COMMAND" ]; then
  # If a custom tensorrt is used, ensure its lib directory is added to the LD_LIBRARY_PATH
  TENSORRT_LIB_PATH="$(dirname "${TENSORRT_COMMAND}")/../../../lib/$(uname -p)-linux-gnu/"
  if ! TENSORRT_LIB_DIR="$(readlink -f "${TENSORRT_LIB_PATH}")"; then
    TENSORRT_LIB_DIR="${TENSORRT_LIB_PATH}"
  fi
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TENSORRT_LIB_DIR}"
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
fi
if [ -z "$ISAAC_ROS_WS" ] && [ -n "$ISAAC_ROS_ASSET_MODEL_PATH" ]; then
  ISAAC_ROS_WS="$(readlink -f "$(dirname "${ISAAC_ROS_ASSET_MODEL_PATH}")/../../../..")"
fi
ASSET_NAME="dnn_stereo_disparity"
VERSION="4.1.0_onnx_trt10.16"
# EULA_URL and ASSET_INSTALL_PATHS are consumed by isaac_ros_asset_eula.sh.
# shellcheck disable=SC2034
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
ASSET_DIR="${MODELS_DIR}/${ASSET_NAME}_v${VERSION#v}"
# shellcheck disable=SC2034
ASSET_INSTALL_PATHS="${ASSET_DIR}/ess.engine ${ASSET_DIR}/light_ess.engine"
ARCHIVE_NAME="dnn_stereo_disparity_v${VERSION#v}.tar.gz"
ESS_MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity/versions/${VERSION}/files/${ARCHIVE_NAME}"

# shellcheck source=/dev/null
source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}"

# Remove old assets dir to prevent errors from tar extraction
rm -rf "${ASSET_DIR}"

isaac_ros_common_download_asset --url "${ESS_MODEL_URL}" --output-path "${MODELS_DIR}/${ARCHIVE_NAME}" --cache-path "${ISAAC_ROS_ESS_MODEL_ARCHIVE}"
ESS_MODEL_DOWNLOAD_RESULT=$?
if [[ -n ${ISAAC_ROS_ASSETS_TEST} ]]; then
  exit ${ESS_MODEL_DOWNLOAD_RESULT}
elif [[ ${ESS_MODEL_DOWNLOAD_RESULT} -ne 0 ]]; then
  echo "ERROR: Failed to download ESS model."
  exit 1
fi

# Extract archive into isaac_ros_assets
tar -xvf "${MODELS_DIR}/${ARCHIVE_NAME}" -C "${MODELS_DIR}"

# Work around a TensorRT 10.16 weakly-typed FP16 auto-cast bug: when a weight
# initializer is shared by multiple nodes, TRT emits a colliding
# "<weight>_output_casted" cast tensor per consumer and the engine build fails
# ("duplicate tensor name ..."). Duplicating shared constants so each consumer
# has its own copy removes the collision; the transform is semantics-preserving.
# Mirrors modelopt's AutoCast graph sanitizer (duplicate_shared_constants).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deduplicate_shared_constants() {
  local onnx_path="$1"
  if [ -x "${SCRIPT_DIR}/deduplicate_shared_constants" ]; then
    # Hermetic py_binary staged alongside this script in the bazel runfiles.
    "${SCRIPT_DIR}/deduplicate_shared_constants" "${onnx_path}"
  else
    # Fallback for non-bazel installs (e.g. colcon `ros2 run`): system python3.
    python3 "${SCRIPT_DIR}/deduplicate_shared_constants.py" "${onnx_path}"
  fi
}
deduplicate_shared_constants "${ASSET_DIR}/ess.onnx"
deduplicate_shared_constants "${ASSET_DIR}/light_ess.onnx"

# Create ESS engine
echo "Converting ESS onnx file to engine file."
${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
  --onnx="${ASSET_DIR}/ess.onnx" \
  --saveEngine="${ASSET_DIR}/ess.engine" \
  --fp16 \
  --staticPlugins="${ASSET_DIR}/plugins/$(uname -m)/ess_plugins.so"

# Create ESS-light engine
echo "Converting ESS light onnx file to engine file."
${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
  --onnx="${ASSET_DIR}/light_ess.onnx" \
  --saveEngine="${ASSET_DIR}/light_ess.engine" \
  --fp16 \
  --staticPlugins="${ASSET_DIR}/plugins/$(uname -m)/ess_plugins.so"
