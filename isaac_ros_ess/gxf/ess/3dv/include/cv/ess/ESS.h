// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_ESS_H_
#define NV_ESS_H_

#include <memory>

#include <cuda_runtime.h>

#include <cv/core/Array.h>
#include <cv/core/Core.h>
#include <cv/core/Model.h>
#include <cv/core/Tensor.h>

namespace cvcore { namespace ess {

/**
 * Describes the algorithm supported for ESS Preprocessing
 */
enum class PreProcessType : uint8_t
{
    RESIZE = 0, // Resize to network dimensions without maintaining aspect ratio
    CENTER_CROP // Crop to network dimensions from center of image
};

/**
 * Describes the parameters for ESS Preprocessing
 */
struct ESSPreProcessorParams
{
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

/*
 * Interface for running pre-processing on ESS model.
 */
class CVCORE_API ESSPreProcessor
{
public:
    /**
   * Default constructor is deleted
   */
    ESSPreProcessor() = delete;

    /**
   * Constructor of ESSPreProcessor.
   * @param preProcessorParams image pre-processing parameters.
   * @param modelInputParams model paramters for network.
   * @param essPreProcessorParams paramaters specific for ess preprocessing.
   */
    ESSPreProcessor(const ImagePreProcessingParams &preProcessorParams, const ModelInputParams &modelInputParams,
                    const ESSPreProcessorParams &essPreProcessorParams);

    /**
   * Destructor of ESSPreProcessor.
   */
    ~ESSPreProcessor();

    /**
   * Main interface to run pre-processing.
   * @param stream cuda stream.
   */

    void execute(Tensor<NCHW, C3, F32> &leftOutput, Tensor<NCHW, C3, F32> &rightOutput,
                 const Tensor<NHWC, C3, U8> &leftInput, const Tensor<NHWC, C3, U8> &rightInput,
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
class CVCORE_API ESS
{
public:
    /**
   * Constructor for ESS.
   * @param imgparams image pre-processing parameters.
   * @param modelInputParams model parameters for network.
   * @param modelInferParams model input inference parameters.
   * @param essPreProcessorParams paramaters specific for ess preprocessing.
   */
    ESS(const ImagePreProcessingParams &imgparams, const ModelInputParams &modelParams,
        const ModelInferenceParams &modelInferParams, const ESSPreProcessorParams &essPreProcessorParams);

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
    void execute(Tensor<NHWC, C1, F32> &disparityMap, const Tensor<NHWC, C3, U8> &leftInput,
                 const Tensor<NHWC, C3, U8> &rightInput, cudaStream_t stream = 0);

    /**
   * Inference function for a given Preprocessed image
   * @param disparityMap Disparity map (CPU/GPU tensor supported)
   * @param leftInput RGB Planar Left image resized to network input dimensions (Only GPU Input Tensor
   * supported)
   * @param rightInput RGB Planar Right image resized to network input dimensions (Only GPU Input Tensor
   * supported)
   * @param stream Cuda stream
   */
    void execute(Tensor<NHWC, C1, F32> &disparityMap, const Tensor<NCHW, C3, F32> &leftInput,
                 const Tensor<NCHW, C3, F32> &rightInput, cudaStream_t stream = 0);

private:
    struct ESSImpl;
    std::unique_ptr<ESSImpl> m_pImpl;
};

/**
 * ESS parameters and implementation
 */
class CVCORE_API ESSPostProcessor
{
public:
    /**
   * Constructor for ESS.
   * @param modelInputParams model parameters for network.
   */
    ESSPostProcessor(const ModelInputParams &modelParams);
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
    void execute(Tensor<NHWC, C1, F32> &outputdisparityMap, const Tensor<NHWC, C1, F32> &inputdisparityMap,
                 cudaStream_t stream = 0);

private:
    struct ESSPostProcessorImpl;
    std::unique_ptr<ESSPostProcessorImpl> m_pImpl;
};

}} // namespace cvcore::ess
#endif
