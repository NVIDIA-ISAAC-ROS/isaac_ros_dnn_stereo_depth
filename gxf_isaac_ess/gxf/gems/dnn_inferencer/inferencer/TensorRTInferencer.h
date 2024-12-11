/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>
#include "Errors.h"
#include "IInferenceBackend.h"
#include "Inferencer.h"

namespace cvcore {
namespace inferencer {

using TensorBase = cvcore::tensor_ops::TensorBase;
class TensorRTInferencer : public IInferenceBackendClient {
 public:
    TensorRTInferencer(const TensorRTInferenceParams& params);

    // Set input layer tensor
    std::error_code setInput(const TensorBase& trtInputBuffer,
        std::string inputLayerName) override;

    // Sets output layer tensor
    std::error_code setOutput(TensorBase& trtOutputBuffer,
        std::string outputLayerName) override;

    // Get the model metadata parsed based on the model file
    // This would be done in initialize call itself. User can access the
    // modelMetaData created using this API.
    ModelMetaData getModelMetaData() const override;

    // Convert onnx mode to engine file
    std::error_code convertModelToEngine(int32_t dla_core,
        const char* model_file, int64_t max_workspace_size, int32_t buildFlags,
        std::size_t max_batch_size);

    // TensorRT will use infer and TensorRT would use enqueueV2
    std::error_code infer(size_t batchSize = 1) override;

    // Applicable only for Native TRT
    std::error_code setCudaStream(cudaStream_t) override;

    // Unregister shared memory for layer
    std::error_code unregister(std::string layerName) override;

    // Unregister all shared memory
    std::error_code unregister() override;

 private:
    ~TensorRTInferencer();
    std::unique_ptr<TRTLogger> m_logger;
    std::unique_ptr<nvinfer1::IRuntime> m_inferenceRuntime;
    size_t m_maxBatchSize;
    std::vector<std::string> m_inputLayers;
    std::vector<std::string> m_outputLayers;
    cudaStream_t m_cudaStream;
    nvinfer1::ICudaEngine* m_inferenceEngine;
    std::unique_ptr<nvinfer1::ICudaEngine> m_ownedInferenceEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_inferenceContext;
    size_t m_ioTensorsCount;
    ModelMetaData m_modelInfo;
    bool m_hasImplicitBatch;
    std::vector<char> m_modelEngineStream;
    size_t m_modelEngineStreamSize = 0;

    std::error_code ParseTRTModel();
    std::error_code getLayerInfo(LayerInfo& layer, std::string layerName);
};

}  // namespace inferencer
}  // namespace cvcore
