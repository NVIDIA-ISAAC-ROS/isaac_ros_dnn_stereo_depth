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

#include "cv/inferencer/Inferencer.h"

#include <iostream>

#include "cv/inferencer/Errors.h"
#include "cv/inferencer/IInferenceBackend.h"
#include "tensorrt/TensorRTInferencer.h"
#include "triton/TritonGrpcInferencer.h"

namespace cvcore { namespace inferencer {
std::mutex InferenceBackendFactory::inferenceMutex;

#ifdef ENABLE_TRITON
std::unordered_map<std::string, std::pair<std::size_t, InferenceBackendClient>>
    InferenceBackendFactory::tritonRemoteMap;

std::error_code InferenceBackendFactory::CreateTritonRemoteInferenceBackendClient(
    InferenceBackendClient &client, const TritonRemoteInferenceParams &params)
{
    std::lock_guard<std::mutex> instanceLock(inferenceMutex);

    if (params.protocolType == BackendProtocol::HTTP)
    {
        return ErrorCode::NOT_IMPLEMENTED;
    }
    std::error_code result = ErrorCode::SUCCESS;
    std::string hashString = params.serverUrl + params.modelName + params.modelVersion;

    try
    {
        if (tritonRemoteMap.find(hashString) != tritonRemoteMap.end())
        {
            client = tritonRemoteMap[hashString].second;
            tritonRemoteMap[hashString].first++;
        }
        else
        {
            tritonRemoteMap[hashString] =
                std::make_pair<std::size_t, InferenceBackendClient>(1, new TritonGrpcInferencer(params));
        }
    }
    catch (std::error_code &e)
    {
        result = e;
    }
    catch (...)
    {
        result = ErrorCode::INVALID_ARGUMENT;
    }
    client = tritonRemoteMap[hashString].second;
    return result;
}

std::error_code InferenceBackendFactory::DestroyTritonRemoteInferenceBackendClient(InferenceBackendClient &client)
{
    std::lock_guard<std::mutex> instanceLock(inferenceMutex);
    for (auto &it : tritonRemoteMap)
    {
        if (it.second.second == client)
        {
            it.second.first--;
            if (it.second.first == 0)
            {
                tritonRemoteMap.erase(it.first);
                client->unregister();
                delete client;
                client = nullptr;
            }
            break;
        }
    }
    client = nullptr;
    return ErrorCode::SUCCESS;
}
#endif

std::error_code InferenceBackendFactory::CreateTensorRTInferenceBackendClient(InferenceBackendClient &client,
                                                                              const TensorRTInferenceParams &params)
{

    std::lock_guard<std::mutex> instanceLock(inferenceMutex);
    std::error_code result = ErrorCode::SUCCESS;
    try
    {
        client = new TensorRTInferencer(params);
    }
    catch (std::error_code &e)
    {
        result = e;
    }
    catch (...)
    {
        result = ErrorCode::INVALID_ARGUMENT;
    }
    return result;
}

std::error_code InferenceBackendFactory::DestroyTensorRTInferenceBackendClient(InferenceBackendClient &client)
{

    std::lock_guard<std::mutex> instanceLock(inferenceMutex);
    if (client != nullptr)
    {
        client->unregister();
        delete client;
        client = nullptr;
    }
    client = nullptr;

    return ErrorCode::SUCCESS;
}

}} // namespace cvcore::inferencer
