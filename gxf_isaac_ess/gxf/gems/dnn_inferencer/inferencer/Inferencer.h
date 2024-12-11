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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "extensions/tensorops/core/Tensor.h"
#include "IInferenceBackend.h"

namespace cvcore {
namespace inferencer {

/**
 * A class to create and destroy a client for a given inference backend type
 */
class InferenceBackendFactory {
 public:
#ifdef ENABLE_TRITON

/**
 * Function to create client for Triton remote inference backend based on the Triton remote
 * inference paramaters.
 * @param client client object created
 * @param Triton remote inference config parameters.
 * @return error code
 */
    static std::error_code CreateTritonRemoteInferenceBackendClient(InferenceBackendClient& client,
        const TritonRemoteInferenceParams&);

/**
 * Function to Destroy the triton grpc client
 * @param client client object created
 * @return error code
 */
    static std::error_code DestroyTritonRemoteInferenceBackendClient(
        InferenceBackendClient& client);
#endif

/**
 * Function to create client for TensorRT inference backend based on the TensorRT
 * inference paramaters.
 * @param client client object created
 * @param TensorRT inference config parameters.
 * @return error code
 */
    static std::error_code CreateTensorRTInferenceBackendClient(InferenceBackendClient& client,
        const TensorRTInferenceParams&);

/**
 * Function to Destroy the tensorrt client
 * @param client client object created
 * @return error code
 */
    static std::error_code DestroyTensorRTInferenceBackendClient(InferenceBackendClient& client);

 private:
#ifdef ENABLE_TRITON
    // Keeps track of any changes in url/model repo path and returns an existing / new client
    // instance.
    static std::unordered_map<std::string,
        std::pair<std::size_t, InferenceBackendClient>> tritonRemoteMap;
#endif
    static std::mutex inferenceMutex;
};

}  // namespace inferencer
}  // namespace cvcore
