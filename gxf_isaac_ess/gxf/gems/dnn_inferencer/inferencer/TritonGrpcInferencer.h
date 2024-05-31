// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifdef ENABLE_TRITON

#include <grpc_client.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "cv/inferencer/Errors.h"
#include "cv/inferencer/IInferenceBackend.h"
#include "cv/inferencer/Inferencer.h"

namespace cvcore {
namespace inferencer {
namespace tc = triton::client;

class TritonGrpcInferencer : public IInferenceBackendClient {
 public:
    TritonGrpcInferencer(const TritonRemoteInferenceParams& params);

    // Set input layer tensor
    std::error_code setInput(const cvcore::TensorBase& trtInputBuffer,
        std::string inputLayerName) override;
    // Sets output layer tensor
    std::error_code setOutput(cvcore::TensorBase& trtOutputBuffer,
        std::string outputLayerName) override;

    // Get the model metadata parsed based on the model file
    // This would be done in initialize call itself. User can access the modelMetaData
    // created using this API.
    ModelMetaData getModelMetaData() const override;

    // Triton will use infer and TensorRT would use enqueueV2
    std::error_code infer(size_t batchSize = 1) override;

    // Applicable only for Native TRT
    std::error_code setCudaStream(cudaStream_t) override;

    // Unregister shared memory for layer
    std::error_code unregister(std::string layerName) override;

    // Unregister all shared memory
    std::error_code unregister() override;

 private:
    // Parse grpc model
    std::error_code ParseGrpcModel();
    std::unique_ptr<triton::client::InferenceServerGrpcClient> client;
    ModelMetaData modelInfo;
    std::vector<std::shared_ptr<tc::InferInput>> inputRequests;
    std::vector<std::shared_ptr<tc::InferRequestedOutput>> outputRequests;
    std::vector<tc::InferInput*> inputMap;
    std::vector<void*> inputMapHistory;
    std::vector<void*> outputMapHistory;
    std::vector<const tc::InferRequestedOutput*> outputMap;
    std::string modelVersion, modelName;
};

}  // namespace inferencer
}  // namespace cvcore
#endif
