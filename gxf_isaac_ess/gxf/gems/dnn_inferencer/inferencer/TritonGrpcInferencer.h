/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */
#pragma once
#ifdef ENABLE_TRITON

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cv/inferencer/Errors.h"
#include "cv/inferencer/IInferenceBackend.h"
#include "cv/inferencer/Inferencer.h"
#include "grpc_client.h"

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
