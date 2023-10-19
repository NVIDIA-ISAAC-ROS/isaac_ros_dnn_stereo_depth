// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include "extensions/tensorops/core/Tensor.h"

namespace cvcore {
namespace inferencer {

/**
 * Struct type to describe input and output layers.
 */
struct LayerInfo {
    size_t index;                 /**< Block Index of layer */
    std::string name;             /**< Name of layer */
    std::vector<int64_t> shape;   /**< Shape of layer */
    cvcore::tensor_ops::ChannelType dataType; /**< Datatype of layer */
    cvcore::tensor_ops::TensorLayout layout;  /**< Tensor layour of layer */
    size_t layerSize;
};

/**
 * Enum class to describe the backend protocol for triton
 */
enum class BackendProtocol {
    GRPC,
    HTTP
};

/**
 * Struct type to describe the input for triton inference.
 */
struct TritonRemoteInferenceParams {
    std::string serverUrl;        /**< Server url created by running the triton server */
    bool verbose;                 /**< Verbose log from backend */
    BackendProtocol protocolType; /**< Backend protocol type */
    std::string modelName;        /**< Model name as per model respoitory */
    std::string modelVersion;     /** Model version as per model repository */
};

/**
 * Struct type to describe the model metadata parsed by the inference backend.
 */
struct ModelMetaData {
    std::string modelName;                                  /**< Model name */
    std::string modelVersion;                               /**< Model version */
    std::unordered_map<std::string, LayerInfo> inputLayers; /**< Map of input layer */
    std::unordered_map<std::string, LayerInfo>
        outputLayers;    /**< Map of output layer information indexed by layer name */
    size_t maxBatchSize; /**< Maximum batch size */
};

/**
 * Enum type for TensorRT inference type
 */
enum class TRTInferenceType {
    TRT_ENGINE,          /**< Inference using TRT engine file */
    TRT_ENGINE_IN_MEMORY /**< Inference using TRT Cuda Engine */
};

/**
 * TRT Logger
 */
class TRTLogger : public nvinfer1::ILogger {
 public:
    void log(Severity severity, const char* msg) noexcept {
        if ((severity == Severity::kERROR) || (severity == Severity::kWARNING)) {
            std::cerr << msg << std::endl;
        }
    }
};

/**
 * Enum class to describe the model build flags
 */
enum OnnxModelBuildFlag {
    NONE = 0,
    kINT8 = 1,
    kFP16 = 2,
    kGPU_FALLBACK = 4,
};

/**
 * Struct type to describe the input for triton inference.
 */
struct TensorRTInferenceParams {
    TRTInferenceType inferType;    /**< TensorRT inference type */
    nvinfer1::ICudaEngine* engine; /**< Cuda engine file for TRT_ENGINE_IN_MEMORY type. */
                                   // Nullptr if TRT_ENGINE is used
    std::string onnxFilePath;      /**< ONNX model file path. */
    std::string engineFilePath;    /**< Engine file path for TRT_ENGINE type. */
    bool force_engine_update;
    int32_t buildFlags;
    int64_t max_workspace_size;
    std::size_t maxBatchSize;      /**< Max Batch size */
    std::vector<std::string> inputLayerNames;  /** Input layer names */
    std::vector<std::string> outputLayerNames; /** Output layer names */
    int32_t dlaID{-1};
};

/**
 * Interface for TritonRemote , Triton C and Native TensorRT inference backend.
 */
class IInferenceBackendClient {
 public:
    virtual ~IInferenceBackendClient() noexcept = default;

/**
 * Function to set input layer data
 * @param trtInputBuffer Input GPU buffer
 * @param inputLayerName Input Layer name
 * @return error code
 */
    virtual std::error_code setInput(const cvcore::tensor_ops::TensorBase& trtInputBuffer,
        std::string inputLayerName) = 0;

/**
 * Function to set output layer data
 * @param trtInputBuffer Output GPU buffer
 * @param outputLayerName Output Layer name
 * @return error code
 */
    virtual std::error_code setOutput(cvcore::tensor_ops::TensorBase& trtOutputBuffer,
        std::string outputLayerName) = 0;

/**
 * Returns the model metadata parsed by the inference backend.
 * @return ModelMetaData
 */
    virtual ModelMetaData getModelMetaData() const = 0;

/**
 * Inference based on input and output set. enqueueV2 for TensorRT and inferSync for Triton.
 * @param Batch size of input for inference. Default set to 1. Must be <= Max Batch Size .
 * @return error code
 */
    virtual std::error_code infer(size_t batchSize = 1) = 0;

/**
 * Sets the cuda stream applicable only for TensorRT backend.
 * @param cudaStream_t cuda input stream
 * @return error code
 */
    virtual std::error_code setCudaStream(cudaStream_t) = 0;

/**
 * Unregisters the tensor mapped from the inference backend
 * @param input/output layer name
 * @return error code
 */
    virtual std::error_code unregister(std::string layerName) = 0;

/**
 * Unregisters all the tensor mappeds from the inference backend
 * @return error code
 */
    virtual std::error_code unregister() = 0;

 protected:
    IInferenceBackendClient()                                = default;
    IInferenceBackendClient(const IInferenceBackendClient&) = default;
    IInferenceBackendClient& operator=(const IInferenceBackendClient&) = default;
    IInferenceBackendClient(IInferenceBackendClient &&) noexcept        = default;
    IInferenceBackendClient& operator=(IInferenceBackendClient &&) noexcept = default;
};

using InferenceBackendClient = IInferenceBackendClient *;

}  // namespace inferencer
}  // namespace cvcore
