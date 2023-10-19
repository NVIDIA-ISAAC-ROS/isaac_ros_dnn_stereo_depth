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
#ifdef ENABLE_TRITON
#include "gems/dnn_inferencer/inferencer/TritonGrpcInferencer.h"
#include <grpc_client.h>
#include <iostream>
#include <string>
#include <vector>
#include "TritonUtils.h"
#include "gems/dnn_inferencer/inferencer/Errors.h"
#include "gems/dnn_inferencer/inferencer/IInferenceBackend.h"
#include "gems/dnn_inferencer/inferencer/Inferencer.h"

namespace cvcore {
namespace inferencer {
namespace tc = triton::client;
using ChannelType = cvcore::tensor_ops::ChannelType;

namespace {
size_t getDataSize(const std::vector<int64_t>& shape, ChannelType dataType) {
    size_t layerShape = 1;
    for (size_t k = 0; k < shape.size(); k++)
        layerShape *= shape[k] <= 0 ? 1 : shape[k];

    return layerShape * GetChannelSize(dataType);
}
}  // namespace

std::error_code TritonGrpcInferencer::ParseGrpcModel() {
    inference::ModelMetadataResponse tritonModelMetadata;
    inference::ModelConfigResponse modelConfig;

    tc::Error err;
    modelInfo.modelName    = modelName;
    modelInfo.modelVersion = modelVersion;
    err                    = client->ModelMetadata(&tritonModelMetadata, modelName, modelVersion);
    err                    = client->ModelConfig(&modelConfig, modelName, modelVersion);
    modelInfo.maxBatchSize = modelConfig.config().max_batch_size();
    bool inputBatchDim     = modelInfo.maxBatchSize > 0;
    for (int i = 0; i < tritonModelMetadata.inputs().size(); i++) {
        LayerInfo layer;
        layer.name       = tritonModelMetadata.inputs(i).name();
        layer.index      = i;
        bool parseStatus = getCVCoreChannelType(layer.dataType,
            tritonModelMetadata.inputs(i).datatype());
        if (!parseStatus) {
            return ErrorCode::INVALID_OPERATION;
        }

        size_t cnt = modelInfo.maxBatchSize == 0 ? 0 : 1;
        if (modelInfo.maxBatchSize != 0)
            layer.shape.push_back(modelInfo.maxBatchSize);
        for (; cnt < tritonModelMetadata.inputs(i).shape().size(); cnt++) {
            layer.shape.push_back(tritonModelMetadata.inputs(i).shape(cnt));
        }
        layer.layerSize                   = getDataSize(layer.shape, layer.dataType);
        modelInfo.inputLayers[layer.name] = layer;
    }
    for (int i = 0; i < tritonModelMetadata.outputs().size(); i++) {
        LayerInfo layer;
        layer.name       = tritonModelMetadata.outputs(i).name();
        layer.index      = i;
        bool parseStatus = getCVCoreChannelType(layer.dataType,
            tritonModelMetadata.inputs(i).datatype());
        if (!parseStatus) {
            return ErrorCode::INVALID_OPERATION;
        }
        layer.layout = TensorLayout::NHWC;
        size_t cnt   = modelInfo.maxBatchSize == 0 ? 0 : 1;
        if (modelInfo.maxBatchSize != 0)
            layer.shape.push_back(modelInfo.maxBatchSize);
        for (; cnt < tritonModelMetadata.outputs(i).shape().size(); cnt++) {
            layer.shape.push_back(tritonModelMetadata.outputs(i).shape(cnt));
        }
        modelInfo.outputLayers[layer.name] = layer;
        layer.layerSize                    = getDataSize(layer.shape, layer.dataType);
    }
    return ErrorCode::SUCCESS;
}

TritonGrpcInferencer::TritonGrpcInferencer(const TritonRemoteInferenceParams& params)
    : modelVersion(params.modelVersion)
    , modelName(params.modelName) {
    tc::Error err = tc::InferenceServerGrpcClient::Create(&client, params.serverUrl,
        params.verbose);

    if (!err.IsOk()) {
        throw make_error_code(InferencerErrorCode::TRITON_SERVER_NOT_READY);
    }

    // Unregistering all shared memory regions for a clean
    // start.
    err = client->UnregisterSystemSharedMemory();
    if (!err.IsOk()) {
        throw make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
    }
    err = client->UnregisterCudaSharedMemory();
    if (!err.IsOk()) {
        throw make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
    }

    ParseGrpcModel();

    // Include the batch dimension if required
    inputRequests.resize(modelInfo.inputLayers.size());
    inputMap.resize(modelInfo.inputLayers.size());
    inputMapHistory.resize(modelInfo.inputLayers.size());
    outputRequests.resize(modelInfo.outputLayers.size());
    outputMap.resize(modelInfo.outputLayers.size());
    outputMapHistory.resize(modelInfo.outputLayers.size());
    for (auto &it : modelInfo.inputLayers) {
        tc::InferInput *inferInputVal;
        std::string tritonDataType;
        bool parseStatus = getTritonChannelType(tritonDataType, it.second.dataType);
        if (!parseStatus) {
            throw make_error_code(InferencerErrorCode::TRITON_REGISTER_LAYER_ERROR);
        }
        err = tc::InferInput::Create(&inferInputVal, it.second.name, it.second.shape,
            tritonDataType);
        if (!err.IsOk()) {
            throw make_error_code(InferencerErrorCode::TRITON_REGISTER_LAYER_ERROR);
        }
        inputRequests[it.second.index].reset(inferInputVal);
    }
    for (auto &it : modelInfo.outputLayers) {
        tc::InferRequestedOutput *output;
        err = tc::InferRequestedOutput::Create(&output, it.second.name);
        if (!err.IsOk()) {
            throw make_error_code(InferencerErrorCode::TRITON_REGISTER_LAYER_ERROR);
        }
        outputRequests[it.second.index].reset(output);
    }
}
cudaError_t CreateCUDAIPCHandle(cudaIpcMemHandle_t* cuda_handle, void* input_d_ptr,
    int deviceId = 0) {
    // Set the GPU device to the desired GPU
    cudaError_t err;
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaIpcGetMemHandle(cuda_handle, input_d_ptr);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaSuccess;
    //  Create IPC handle for data on the gpu
}

// Set input layer tensor
std::error_code TritonGrpcInferencer::setInput(const cvcore::TensorBase& trtInputBuffer,
    std::string inputLayerName) {
    if (trtInputBuffer.isCPU()) {
        return ErrorCode::INVALID_ARGUMENT;
    }

    size_t index = modelInfo.inputLayers[inputLayerName].index;
    if (inputMapHistory[index] != reinterpret_cast<void *>(trtInputBuffer.getData())) {
        inputMapHistory[index] = trtInputBuffer.getData();
        unregister(inputLayerName);
        cudaIpcMemHandle_t input_cuda_handle;
        cudaError_t cudaStatus = CreateCUDAIPCHandle(&input_cuda_handle,
            reinterpret_cast<void *>(trtInputBuffer.getData()));
        if (cudaStatus != cudaSuccess) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
        }

        tc::Error err;
        err = client->RegisterCudaSharedMemory(inputLayerName.c_str(), input_cuda_handle, 0,
                                               trtInputBuffer.getDataSize());
        if (!err.IsOk()) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
        }

        size_t index = modelInfo.inputLayers[inputLayerName].index;
        err = inputRequests[index]->SetSharedMemory(inputLayerName.c_str(),
            trtInputBuffer.getDataSize(), 0);
        inputMap[index] = inputRequests[index].get();
        if (!err.IsOk()) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
            err = inputRequests[index]->SetSharedMemory(inputLayerName.c_str(),
                trtInputBuffer.getDataSize(), 0);
            inputMap[index] = inputRequests[index].get();
            if (!err.IsOk()) {
                return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
            }
        }
    }
    return ErrorCode::SUCCESS;
}

// Sets output layer tensor
std::error_code TritonGrpcInferencer::setOutput(cvcore::TensorBase& trtOutputBuffer,
    std::string outputLayerName) {
    if (trtOutputBuffer.isCPU()) {
        return ErrorCode::INVALID_ARGUMENT;
    }

    size_t index = modelInfo.outputLayers[outputLayerName].index;
    if (outputMapHistory[index] != reinterpret_cast<void *>(trtOutputBuffer.getData())) {
        outputMapHistory[index] = trtOutputBuffer.getData();
        unregister(outputLayerName);
        cudaIpcMemHandle_t outputCudaHandle;
        CreateCUDAIPCHandle(&outputCudaHandle,
            reinterpret_cast<void *>(trtOutputBuffer.getData()));
        tc::Error err;
        err = client->RegisterCudaSharedMemory(outputLayerName.c_str(), outputCudaHandle, 0,
                                               trtOutputBuffer.getDataSize());
        if (!err.IsOk()) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
        }

        err = outputRequests[index]->SetSharedMemory(outputLayerName.c_str(),
            trtOutputBuffer.getDataSize(), 0);
        if (!err.IsOk()) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
        }
        outputMap[index] = outputRequests[index].get();
    }
    return ErrorCode::SUCCESS;
}

// Get the model metadata parsed based on the model file
// This would be done in initialize call itself. User can access the modelMetaData created
// using this API.
ModelMetaData TritonGrpcInferencer::getModelMetaData() const {
    return modelInfo;
}

// Triton will use infer and TensorRT would use enqueueV2
std::error_code TritonGrpcInferencer::infer(size_t batchSize) {
    tc::InferResult *results;
    tc::Headers httpHeaders;
    tc::InferOptions options(modelInfo.modelName);
    options.model_version_ = modelInfo.modelVersion;
    for (auto &inputLayer : modelInfo.inputLayers) {
        LayerInfo inputLayerInfo = inputLayer.second;
        size_t index             = inputLayerInfo.index;
        tc::Error err;
        err = inputRequests[index]->SetSharedMemory(inputLayerInfo.name.c_str(),
            inputLayerInfo.layerSize * batchSize, 0);
        if (!err.IsOk()) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
        }
    }
    for (auto &outputLayer : modelInfo.outputLayers) {
        LayerInfo outputLayerInfo = outputLayer.second;
        size_t index              = outputLayerInfo.index;
        tc::Error err;
        err = outputRequests[index]->SetSharedMemory(outputLayerInfo.name.c_str(),
            outputLayerInfo.layerSize * batchSize, 0);
        if (!err.IsOk()) {
            return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
        }
    }
    tc::Error err = client->Infer(&results, options, inputMap, outputMap, httpHeaders);
    if (!err.IsOk()) {
        return make_error_code(InferencerErrorCode::TRITON_INFERENCE_ERROR);
    }

    return ErrorCode::SUCCESS;
}

// Applicable only for Native TRT
std::error_code TritonGrpcInferencer::setCudaStream(cudaStream_t) {
    return ErrorCode::INVALID_OPERATION;
}

std::error_code TritonGrpcInferencer::unregister(std::string layerName) {
    tc::Error err;
    inference::CudaSharedMemoryStatusResponse status;
    err = client->CudaSharedMemoryStatus(&status);
    if (!err.IsOk()) {
        return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
    }
    err = client->UnregisterCudaSharedMemory(layerName.c_str());
    if (!err.IsOk()) {
        return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
    }
    return ErrorCode::SUCCESS;
}

std::error_code TritonGrpcInferencer::unregister() {
    tc::Error err;
    inference::CudaSharedMemoryStatusResponse status;
    err = client->CudaSharedMemoryStatus(&status);
    if (!err.IsOk()) {
        return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
    }
    err = client->UnregisterCudaSharedMemory();
    if (!err.IsOk()) {
        return make_error_code(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR);
    }
    return ErrorCode::SUCCESS;
}
}  // namespace inferencer
}  // namespace cvcore
#endif
