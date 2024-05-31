#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

ASSET_NAME="dnn_stereo_disparity"
VERSION="4.0.0"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
ASSET_DIR="${MODELS_DIR}/${ASSET_NAME}_v${VERSION}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/ess.engine ${ASSET_DIR}/light_ess.engine"
ARCHIVE_NAME="dnn_stereo_disparity_v4.0.0.tar.gz"
ESS_MODEL_URL="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/isaac/dnn_stereo_disparity/${VERSION}/files?redirect=true&path=${ARCHIVE_NAME}"

source "isaac_ros_asset_eula.sh"

# Skip if ESS engine files already exist
ASSET_INSTALL_PATHS_LIST=($ASSET_INSTALL_PATHS)
if [[ -f ${ASSET_INSTALL_PATHS_LIST[0]} ]] && [[ -f ${ASSET_INSTALL_PATHS_LIST[1]} ]]
then
    echo "ESS model engine files already exist at \"${ASSET_DIR}\". Skipping download."
    exit 0
fi

# Download and extract model archive
echo "Downloading ESS etlt file."
wget "${ESS_MODEL_URL}" -O "${MODELS_DIR}/${ARCHIVE_NAME}"
tar -xvf "${MODELS_DIR}/${ARCHIVE_NAME}" -C "${MODELS_DIR}"

# Create ESS engine
echo "Converting ESS etlt file to engine file."
export LD_PRELOAD="${ASSET_DIR}/plugins/$(uname -m)/ess_plugins.so"
tao-converter \
    -k ess \
    -t fp16 \
    -e "${ASSET_DIR}/ess.engine" \
    -o output_left,output_conf \
    "${ASSET_DIR}/ess.etlt"

# Create ESS-light engine
echo "Converting ESS light etlt file to engine file."
tao-converter \
    -k ess \
    -t fp16 \
    -e "${ASSET_DIR}/light_ess.engine" \
    -o output_left,output_conf \
    "${ASSET_DIR}/light_ess.etlt"

