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

#ifndef CVCORE_INFERENCER_H
#define CVCORE_INFERENCER_H

#include <memory>
#include <mutex>
#include <unordered_map>

#include "cv/core/Tensor.h"
#include "cv/inferencer/IInferenceBackend.h"

namespace cvcore { namespace inferencer {

/**
 * A class to create and destroy a client for a given inference backend type
 */
class InferenceBackendFactory
{
public:
#ifdef ENABLE_TRITON

/**
 * Function to create client for Triton remote inference backend based on the Triton remote inference paramaters.
 * @param client client object created
 * @param Triton remote inference config parameters.
 * @return error code
 */
    static std::error_code CreateTritonRemoteInferenceBackendClient(InferenceBackendClient &client,
                                                                    const TritonRemoteInferenceParams &);

/**
 * Function to Destroy the triton grpc client
 * @param client client object created
 * @return error code
 */
    static std::error_code DestroyTritonRemoteInferenceBackendClient(InferenceBackendClient &client);
#endif

/**
 * Function to create client for TensorRT inference backend based on the TensorRT inference paramaters.
 * @param client client object created
 * @param TensorRT inference config parameters.
 * @return error code
 */
    static std::error_code CreateTensorRTInferenceBackendClient(InferenceBackendClient &client,
                                                                const TensorRTInferenceParams &);

/**
 * Function to Destroy the tensorrt client
 * @param client client object created
 * @return error code
 */
    static std::error_code DestroyTensorRTInferenceBackendClient(InferenceBackendClient &client);

private:
#ifdef ENABLE_TRITON
    // Keeps track of any changes in url/model repo path and returns an existing / new client instance.
    static std::unordered_map<std::string, std::pair<std::size_t, InferenceBackendClient>> tritonRemoteMap;
#endif
    static std::mutex inferenceMutex;
};
}} // namespace cvcore::inferencer
#endif
