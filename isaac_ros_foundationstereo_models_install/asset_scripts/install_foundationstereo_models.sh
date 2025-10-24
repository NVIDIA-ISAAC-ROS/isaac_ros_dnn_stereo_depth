#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Download and tao-convert FoundationStereo models.
# * Models will be stored in the isaac_ros_assets dir
# * The script must be called with the --eula argument prior to downloading.

set -e

# Available model configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS["low_res"]="320x736:deployable_foundationstereo_small_320x736_v1.0.onnx"
MODEL_CONFIGS["high_res"]="576x960:deployable_foundationstereo_small_576x960_v1.0.onnx"

# Default model - use environment variable if set and valid, otherwise use hardcoded default
if [[ -n "$FOUNDATIONSTEREO_MODEL_RES" ]] && [[ -n "${MODEL_CONFIGS[$FOUNDATIONSTEREO_MODEL_RES]}" ]]; then
    SELECTED_MODEL="$FOUNDATIONSTEREO_MODEL_RES"
elif [[ -n "$FOUNDATIONSTEREO_MODEL_RES" ]]; then
    echo "Warning: Invalid FOUNDATIONSTEREO_MODEL_RES='$FOUNDATIONSTEREO_MODEL_RES'. Using default 'high_res'. Available resolutions: ${!MODEL_CONFIGS[*]}" >&2
    SELECTED_MODEL="high_res"
else
    SELECTED_MODEL="high_res"
fi

function usage() {
    echo "Usage: $0 --eula [--model_res|-m <model_resolution>]"
    echo "Available resolutions:"
    echo "  low_res: 320x736 resolution"
    echo "  high_res: 576x960 resolution (default)"
    echo ""
    echo "Environment Variables:"
    echo "  FOUNDATIONSTEREO_MODEL_RES: Set default model resolution (low_res|high_res)"
}

PASSTHRU_ARGS=()
while (( "$#" )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--model_res)
            if [[ -n "$2" ]]; then
                SELECTED_MODEL="$2"; shift 2
            else
                echo "Error: --model_res requires a value" >&2; exit 2
            fi
            ;;
        *)
            PASSTHRU_ARGS+=("$1"); shift
            ;;
    esac
done

# Validate model selection
if [[ -z "${MODEL_CONFIGS[$SELECTED_MODEL]}" ]]; then
    echo "Error: Invalid resolution '$SELECTED_MODEL'. Available resolutions: ${!MODEL_CONFIGS[*]}" >&2
    exit 2
fi

# Parse selected model configuration
IFS=':' read -r RESOLUTION MODEL_FILE_NAME <<< "${MODEL_CONFIGS[$SELECTED_MODEL]}"
IFS='x' read -r HEIGHT WIDTH <<< "$RESOLUTION"

ASSET_NAME="foundationstereo"
VERSION="deployable_foundation_stereo_small_v1.0"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
ASSET_DIR="${MODELS_DIR}/${VERSION}"
FOUNDATIONSTEREO_MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationstereo/versions/${VERSION}/files/${MODEL_FILE_NAME}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/foundationstereo_${RESOLUTION}.engine"

# Pass arguments to the EULA helper
set -- "${PASSTHRU_ARGS[@]}"
source "isaac_ros_asset_eula.sh"

# Create directories if they don't exist
mkdir -p ${ASSET_DIR}

# Download ONNX model file
echo "Downloading FoundationStereo ${SELECTED_MODEL} model (${RESOLUTION} resolution)."
wget "${FOUNDATIONSTEREO_MODEL_URL}" -O "${ASSET_DIR}/foundationstereo_${RESOLUTION}.onnx"

# Create FoundationStereo engine using TensorRT
echo "Converting FoundationStereo ${SELECTED_MODEL} onnx file to engine file."
/usr/src/tensorrt/bin/trtexec \
    --onnx=${ASSET_DIR}/foundationstereo_${RESOLUTION}.onnx \
    --saveEngine=${ASSET_DIR}/foundationstereo_${RESOLUTION}.engine \
    --fp16
