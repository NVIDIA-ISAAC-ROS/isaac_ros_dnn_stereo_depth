// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCORE_TRTBACKEND_H
#define CVCORE_TRTBACKEND_H

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace cvcore {

/**
 * Enum to specify different precision types.
 */
enum TRTBackendPrecision
{
    INT8, /**< 8-bit integer precision. */
    FP16, /**< 16-bit float precision. */
    FP32  /**< 32-bit float precision. */
};

/**
 * Enum to specify TensorRT blob layout type.
 */
enum TRTBackendBlobLayout
{
    PLANAR,     /**< planar input type. */
    INTERLEAVED /**< interleaved input type. */
};

/**
 * Struct to store TensorRT blob dimensions.
 */
struct TRTBackendBlobSize
{
    int channels; /**< channels count. */
    int height;   /**< blob height. */
    int width;    /**< blob width. */
};

/**
 * Enum to specify model type.
 */
enum ModelType
{
    ONNX,                /**< ONNX model. */
    UFF,                 /**< Uff model. */
    TRT_ENGINE,          /**< serialized TensorRT engine. */
    TRT_ENGINE_IN_MEMORY /**< a memory pointer to deserialized TensorRT ICudaEngine. */
};

/**
 * Parameters for TRTBackend.
 */
struct TRTBackendParams
{
    // This constructor is for backward compatibility with all the other core modules which use trtbackend.
    TRTBackendParams(std::vector<std::string> inputLayers, std::vector<TRTBackendBlobSize> inputDims,
                     std::vector<std::string> outputLayers, std::string weightsPath, std::string enginePath,
                     TRTBackendPrecision precision, int batchSize, ModelType modelType, bool explicitBatch,
                     void *engine = nullptr, TRTBackendBlobLayout layout = PLANAR)
        : inputLayers(inputLayers)
        , inputDims(inputDims)
        , outputLayers(outputLayers)
        , weightsPath(weightsPath)
        , enginePath(enginePath)
        , precision(precision)
        , batchSize(batchSize)
        , modelType(modelType)
        , explicitBatch(explicitBatch)
        , trtEngineInMemory(engine)
        , inputLayout(layout)
    {
    }

    std::vector<std::string> inputLayers;      /**< names of input layers. */
    std::vector<TRTBackendBlobSize> inputDims; /**< dimensions of input layers. */
    std::vector<std::string> outputLayers;     /**< names of output layers. */
    std::string weightsPath;                   /**< model weight path. */
    std::string enginePath;                    /**< TensorRT engine path. */
    TRTBackendPrecision precision;             /**< TensorRT engine precision. */
    int batchSize;                             /**< inference batch size. */
    ModelType modelType;                       /**< model type. */
    bool explicitBatch;                        /**< whether it is explicit batch dimension. */
    void *
        trtEngineInMemory; /**< pointer to hold deserialized TensorRT ICudaEngine, for ModelType::TRT_ENGINE_IN_MEMORY. */
    TRTBackendBlobLayout inputLayout; /**< input blob layout. */
};

// Forward declaration
struct TRTImpl;

/**
 * TensorRT wrapper class.
 */
class TRTBackend
{
public:
    /**
     * Constructor of TRTBackend.
     * @param modelFilePath path of the network model.
     * @param precision TensorRT precision type.
     */
    TRTBackend(const char *modelFilePath, TRTBackendPrecision precision, int batchSize = 1, bool explicitBatch = false);

    /**
     * Constructor of TRTBackend.
     * @param inputParams parameters of TRTBackend.
     */
    TRTBackend(TRTBackendParams &inputParams);

    /**
     * Destructor of TRTBackend.
     */
    ~TRTBackend();

    /**
     * Run inference.
     * @param buffer input GPU buffers.
     */
    [[deprecated]] void infer(void **buffer);

    /**
     * Run inference.
     * @param buffer input GPU buffers.
     * @param batchSize run infer with specific batch size, passed in setBindingDimension() call.
     * @param stream update cuda stream in this instance.
     */
    void infer(void **buffer, int batchSize, cudaStream_t stream);

    /**
     * Get the cuda stream for TRTBackend.
     * @return return cuda stream ID.
     */
    [[deprecated]] cudaStream_t getCUDAStream() const;

    /**
     * Set the cuda stream for TRTBackend.
     * @param stream specified cudaStream_t for the TensorRT Engine.
     */
    [[deprecated]] void setCUDAStream(cudaStream_t stream);

    /**
     * Get all input/output bindings count.
     * @return number of all bindings.
     */
    int getBlobCount() const;

    /**
     * Get the blob dimension for given blob index.
     * @param blobIndex blob index.
     * @return blob dimension for the given index.
     */
    TRTBackendBlobSize getTRTBackendBlobSize(int blobIndex) const;

    /**
     * Get the total number of elements for the given blob index.
     * @param blobIndex blob index.
     * @return total size for the given index.
     */
    int getBlobLinearSize(int blobIndex) const;

    /**
     * Get the blob index for the given blob name.
     * @param blobName blob name.
     * @return blob index for the given name.
     */
    int getBlobIndex(const char *blobName) const;

    /**
     * Check if binding is input.
     * @param index binding index.
     * @return whether the binding is input.
     */
    bool bindingIsInput(const int index) const;

private:
    // TRT related variables
    std::unique_ptr<TRTImpl> m_pImpl;
};

} // namespace cvcore

#endif // CVCORE_TRTBACKEND_H
