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
  # If a custom tensorrt is used, ensure it's lib directory is added to the LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(readlink -f $(dirname ${TENSORRT_COMMAND})/../../../lib/x86_64-linux-gnu/)"
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
fi
if [ -z "$ISAAC_ROS_WS" ] && [ -n "$ISAAC_ROS_ASSET_MODEL_PATH" ]; then
  ISAAC_ROS_WS="$(readlink -f $(dirname ${ISAAC_ROS_ASSET_MODEL_PATH})/../../../..)"
fi
ASSET_NAME="dnn_stereo_disparity"
VERSION="4.1.0_onnx_trt10.13"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
ASSET_DIR="${MODELS_DIR}/${ASSET_NAME}_v${VERSION#v}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/ess.engine ${ASSET_DIR}/light_ess.engine"
ARCHIVE_NAME="dnn_stereo_disparity_v${VERSION#v}.tar.gz"
ESS_MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity/versions/${VERSION}/files/${ARCHIVE_NAME}"

source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}"

# Download and extract model archive
echo "Downloading ESS onnx file."
wget -nv "${ESS_MODEL_URL}" -O "${MODELS_DIR}/${ARCHIVE_NAME}"
tar -xvf "${MODELS_DIR}/${ARCHIVE_NAME}" -C "${MODELS_DIR}"

# Create ESS engine
echo "Converting ESS onnx file to engine file."
${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --onnx=${ASSET_DIR}/ess.onnx \
    --saveEngine=${ASSET_DIR}/ess.engine \
    --fp16 \
    --staticPlugins=${ASSET_DIR}/plugins/$(uname -m)/ess_plugins.so

# Create ESS-light engine
echo "Converting ESS light onnx file to engine file."
${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --onnx=${ASSET_DIR}/light_ess.onnx \
    --saveEngine=${ASSET_DIR}/light_ess.engine \
    --fp16 \
    --staticPlugins=${ASSET_DIR}/plugins/$(uname -m)/ess_plugins.so
