// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "extensions/tensorops/core/Array.h"
#include "extensions/tensorops/core/Core.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/Tensor.h"

namespace nvidia {
namespace isaac {
namespace ess {

using TensorLayout = cvcore::tensor_ops::TensorLayout;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using TensorDimension = cvcore::tensor_ops::TensorDimension;
using Tensor_NHWC_C3_U8 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC,
    ChannelCount::C3, ChannelType::U8>;
using Tensor_NHWC_C3_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC,
    ChannelCount::C3, ChannelType::F32>;
using Tensor_NHWC_C1_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC,
    ChannelCount::C1, ChannelType::F32>;
using Tensor_NCHW_C3_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NCHW,
    ChannelCount::C3, ChannelType::F32>;
using Tensor_HWC_C3_U8 = cvcore::tensor_ops::Tensor<TensorLayout::HWC,
    ChannelCount::C3, ChannelType::U8>;
using DisparityLevels = cvcore::tensor_ops::Array<int>;
using TensorBase = cvcore::tensor_ops::TensorBase;
using ImagePreProcessingParams = cvcore::tensor_ops::ImagePreProcessingParams;
using ImageType = cvcore::tensor_ops::ImageType;

/**
 * Struct to describe input type required by the model
 */
struct ModelInputParams {
    size_t maxBatchSize;      /**< maxbatchSize supported by network*/
    cvcore::tensor_ops::ImageType modelInputType; /**< Input Layout type */
};

/**
 * Struct to describe parameters used for building engine file
 */
struct ModelBuildParams {
    bool force_engine_update;
    std::string onnx_file_path;
    bool enable_fp16;
    int64_t max_workspace_size;
    int64_t dla_core;
};

/**
 * Struct to describe the model
 */
struct ModelInferenceParams {
    std::string engineFilePath;             /**< Engine file path. */
    std::vector<std::string> inputLayers;   /**< names of input layers. */
    std::vector<std::string> outputLayers;  /**< names of output layers. */
};

/**
 * Describes the algorithm supported for ESS Preprocessing
 */
enum class PreProcessType : uint8_t {
    RESIZE = 0,  // Resize to network dimensions without maintaining aspect ratio
    CENTER_CROP  // Crop to network dimensions from center of image
};

/**
 * Describes the parameters for ESS Preprocessing
 */
struct ESSPreProcessorParams {
    /* Preprocessing steps for ESS */
    PreProcessType preProcessType;
};

/**
 *  Default parameters for the preprocessing pipeline.
 */
CVCORE_API extern const ImagePreProcessingParams defaultPreProcessorParams;

/**
 *  Default parameters to describe the input expected for the model.
 */
CVCORE_API extern const ModelInputParams defaultModelInputParams;

/**
 *  Default parameters to describe the model inference parameters.
 */
CVCORE_API extern const ModelInferenceParams defaultInferenceParams;

/**
 *  Default parameters for the ESS Preprocessing
 */
CVCORE_API extern const ESSPreProcessorParams defaultESSPreProcessorParams;

/**
 *  Version of inference engine used
 */
CVCORE_API extern const int InferencerVersion;

/*
 * Interface for running pre-processing on ESS model.
 */
class CVCORE_API ESSPreProcessor {
 public:
    /**
     * Default constructor is deleted
     */
    ESSPreProcessor() = delete;

    /**
     * Constructor of ESSPreProcessor.
     * @param preProcessorParams image pre-processing parameters.
     * @param modelInputParams model paramters for network.
     * @param output_width PreProcessor output width.
     * @param output_height PreProcessor output height.
     * @param essPreProcessorParams paramaters specific for ess preprocessing.
     */
    ESSPreProcessor(const ImagePreProcessingParams & preProcessorParams,
        const ModelInputParams & modelInputParams,
        size_t output_width, size_t output_height,
        const ESSPreProcessorParams & essPreProcessorParams);

    /**
     * Destructor of ESSPreProcessor.
     */
    ~ESSPreProcessor();

    /**
     * Main interface to run pre-processing.
     * @param stream cuda stream.
     */
    void execute(Tensor_NCHW_C3_F32& leftOutput, Tensor_NCHW_C3_F32& rightOutput,
                 const Tensor_NHWC_C3_U8& leftInput, const Tensor_NHWC_C3_U8& rightInput,
                 cudaStream_t stream = 0);

 private:
    /**
     * Implementation of ESSPreProcessor.
     */
    struct ESSPreProcessorImpl;
    std::unique_ptr<ESSPreProcessorImpl> m_pImpl;
};

/**
 * ESS parameters and implementation
 */
class CVCORE_API ESS {
 public:
    /**
     * Constructor for ESS.
     * @param imgparams image pre-processing parameters.
     * @param modelInputParams model paramters for network.
     * @param modelBuildParams model parameters for building engine.
     * @param modelInferParams model input inference parameters.
     * @param essPreProcessorParams paramaters specific for ess preprocessing.
     */
    ESS(const ImagePreProcessingParams & imgparams,
        const ModelInputParams & modelInputParams,
        const ModelBuildParams & modelBuildParams,
        const ModelInferenceParams & modelInferParams,
        const ESSPreProcessorParams & essPreProcessorParams);

    /**
     * Default constructor not supported.
     */
    ESS() = delete;

    /**
     * Destructor for ESS.
     */
    ~ESS();

    /**
     * Inference function for a given BGR image
     * @param disparityMap Disparity map (CPU/GPU tensor supported)
     * @param leftInput RGB/BGR Interleaved Left image (Only GPU Input Tensor
     * supported)
     * @param rightInput RGB/BGR Interleaved Right image (Only GPU Input Tensor
     * supported)
     * @param stream Cuda stream
     */
    void execute(Tensor_NHWC_C1_F32 & disparityMap, Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NHWC_C3_U8 & leftInput, const Tensor_NHWC_C3_U8 & rightInput,
                 cudaStream_t stream = 0);

    /**
     * Inference function for a given Preprocessed image
     * @param disparityMap Disparity map (CPU/GPU tensor supported)
     * @param leftInput RGB Planar Left image resized to network input dimensions (Only GPU Input Tensor
     * supported)
     * @param rightInput RGB Planar Right image resized to network input dimensions (Only GPU Input Tensor
     * supported)
     * @param stream Cuda stream
     */
    void execute(Tensor_NHWC_C1_F32& disparityMap,  Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NCHW_C3_F32& leftInput, const Tensor_NCHW_C3_F32& rightInput,
                 cudaStream_t stream = 0);

    /**
     * Helper function to get Model output Height
     */
    size_t getModelOutputHeight();

    /**
     * Helper function to get Model output Width
     */
    size_t getModelOutputWidth();

 private:
    struct ESSImpl;
    std::unique_ptr<ESSImpl> m_pImpl;
};

/**
 * ESS parameters and implementation
 */
class CVCORE_API ESSPostProcessor {
 public:
    /**
     * Constructor for ESS.
     * @param modelInputParams model parameters for network.
     * @param output_width PostProcessor output width.
     * @param output_height PostProcessor output height.
     */
    ESSPostProcessor(const ModelInputParams & modelParams,
                     size_t output_width, size_t output_height);
    /**
     * Default constructor not supported.
     */
    ESSPostProcessor() = delete;

    /**
     * Destructor for ESS.
     */
    ~ESSPostProcessor();

    /**
     * Inference function for a given BGR image
     * @param outputdisparityMap Disparity map rescaled to orginal resolution (CPU/GPU tensor)
     * @param inputDisparityMap input Disparity map (GPU tensor)
     * @param stream Cuda stream
     */
    void execute(Tensor_NHWC_C1_F32 & outputdisparityMap,
        const Tensor_NHWC_C1_F32 & inputdisparityMap,
        cudaStream_t stream = 0);

 private:
    struct ESSPostProcessorImpl;
    std::unique_ptr<ESSPostProcessorImpl> m_pImpl;
};

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia
