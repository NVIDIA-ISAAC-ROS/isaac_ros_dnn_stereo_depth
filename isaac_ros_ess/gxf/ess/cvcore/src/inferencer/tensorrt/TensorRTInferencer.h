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

#ifndef TENSORRT_INFERENCER_H
#define TENSORRT_INFERENCER_H

#include <iostream>

#include <NvInferRuntimeCommon.h>
#include "cv/inferencer/Errors.h"
#include "cv/inferencer/IInferenceBackend.h"
#include "cv/inferencer/Inferencer.h"

namespace cvcore { namespace inferencer {

class TensorRTInferencer : public IInferenceBackendClient
{
public:
    TensorRTInferencer(const TensorRTInferenceParams &params);

    // Set input layer tensor
    virtual std::error_code setInput(const cvcore::TensorBase &trtInputBuffer, std::string inputLayerName) override;

    // Sets output layer tensor
    virtual std::error_code setOutput(cvcore::TensorBase &trtOutputBuffer, std::string outputLayerName) override;

    // Get the model metadata parsed based on the model file
    // This would be done in initialize call itself. User can access the modelMetaData created using this API.
    virtual ModelMetaData getModelMetaData() const override;

    // TensorRT will use infer and TensorRT would use enqueueV2
    virtual std::error_code infer(size_t batchSize = 1) override;

    // Applicable only for Native TRT
    virtual std::error_code setCudaStream(cudaStream_t) override; // Only in TRT

    // Unregister shared memory for layer
    virtual std::error_code unregister(std::string layerName) override;

    // Unregister all shared memory
    virtual std::error_code unregister() override;

private:
    ~TensorRTInferencer();
    std::unique_ptr<TRTLogger> m_logger;
    std::unique_ptr<nvinfer1::IRuntime> m_inferenceRuntime;
    size_t m_maxBatchSize;
    std::vector<std::string> m_inputLayers;
    std::vector<std::string> m_outputLayers;
    cudaStream_t m_cudaStream;
    nvinfer1::ICudaEngine *m_inferenceEngine;
    std::unique_ptr<nvinfer1::ICudaEngine> m_ownedInferenceEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_inferenceContext;
    size_t m_bindingsCount;
    ModelMetaData m_modelInfo;
    std::vector<void *> m_buffers;
    bool m_hasImplicitBatch;

    std::error_code ParseTRTModel();
    std::error_code getLayerInfo(LayerInfo &layer, std::string layerName);
};

}} // namespace cvcore::inferencer
#endif
