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
#include "gems/dnn_inferencer/inferencer/TensorRTInferencer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gems/dnn_inferencer/inferencer/Errors.h"
#include "gems/dnn_inferencer/inferencer/IInferenceBackend.h"
#include "gems/dnn_inferencer/inferencer/Inferencer.h"
#include "gems/dnn_inferencer/inferencer/TensorRTUtils.h"
#include "gxf/core/expected.hpp"
#include <NvOnnxParser.h>

namespace cvcore {
namespace inferencer {

namespace {
//  using namespace cvcore::tensor_ops;
using TensorBase = cvcore::tensor_ops::TensorBase;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using ErrorCode = cvcore::tensor_ops::ErrorCode;
size_t getDataSize(const std::vector<int64_t>& shape, ChannelType dataType) {
    size_t layerShape = 1;
    for (size_t k = 0; k < shape.size(); k++)
        layerShape *= shape[k] <= 0 ? 1 : shape[k];

    return layerShape * GetChannelSize(dataType);
}
}  // namespace

std::error_code TensorRTInferencer::getLayerInfo(LayerInfo& layer, std::string layerName) {
    layer.name = layerName;
    layer.index = m_inferenceEngine->getBindingIndex(layerName.c_str());
    auto dim = m_inferenceEngine->getBindingDimensions(layer.index);
    nvinfer1::TensorFormat tensorFormat = m_inferenceEngine->getBindingFormat(layer.index);

    std::error_code err;
    err = getCVCoreChannelLayoutFromTensorRT(layer.layout, tensorFormat);
    if (err != make_error_code(ErrorCode::SUCCESS)) {
        return ErrorCode::INVALID_ARGUMENT;
    }

    for (int32_t cnt = 0; cnt < dim.nbDims; cnt++) {
        layer.shape.push_back(dim.d[cnt]);
    }

    err = getCVCoreChannelTypeFromTensorRT(layer.dataType,
        m_inferenceEngine->getBindingDataType(layer.index));
    layer.layerSize = getDataSize(layer.shape, layer.dataType);
    if (err != make_error_code(ErrorCode::SUCCESS)) {
        return ErrorCode::INVALID_ARGUMENT;
    }

    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::ParseTRTModel() {
    m_modelInfo.modelName    = m_inferenceEngine->getName();
    m_modelInfo.modelVersion = "";
    m_modelInfo.maxBatchSize = m_maxBatchSize;
    std::error_code err;
    for (size_t i = 0; i < m_inputLayers.size(); i++) {
        LayerInfo layer;
        err = getLayerInfo(layer, m_inputLayers[i]);
        if (err != make_error_code(ErrorCode::SUCCESS)) {
            return err;
        }
        m_modelInfo.inputLayers[layer.name] = layer;
    }
    for (size_t i = 0; i < m_outputLayers.size(); i++) {
        LayerInfo layer;
        err = getLayerInfo(layer, m_outputLayers[i]);
        if (err != make_error_code(ErrorCode::SUCCESS)) {
            return err;
        }
        m_modelInfo.outputLayers[layer.name] = layer;
    }

    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::convertModelToEngine(int32_t dla_core,
  const char* model_file, int64_t max_workspace_size, int32_t buildFlags,
  std::size_t max_batch_size) {
  GXF_LOG_INFO("Convert to engine from onnx file: %s", model_file);
  // Creates the engine Builder
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*(m_logger.get())));

  // Builder Config provides options to the Builder
  std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig(builder->createBuilderConfig());
  builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size);

  if (dla_core >= 0) {
    builderConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    builderConfig->setDLACore(dla_core);
  }
  if (buildFlags & kINT8) {
    builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
    builderConfig->setInt8Calibrator(nullptr);
  }
  if (buildFlags & OnnxModelBuildFlag::kFP16) {
    builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  if (buildFlags & kGPU_FALLBACK) {
    builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  }

  // Parses ONNX with explicit batch size for support of dynamic shapes/batch
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  std::unique_ptr<nvonnxparser::IParser> onnx_parser(
      nvonnxparser::createParser(*network, *(m_logger.get())));
  if (!onnx_parser->parseFromFile(model_file,
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    GXF_LOG_ERROR("Failed to parse ONNX file %s", model_file);
    return ErrorCode::INVALID_ARGUMENT;
  }

  // Provides optimization profile for dynamic size input bindings
  nvinfer1::IOptimizationProfile* optimization_profile = builder->createOptimizationProfile();
  // Checks input dimensions and adds to optimization profile if needed
  const int number_inputs = network->getNbInputs();
  for (int i = 0; i < number_inputs; ++i) {
    auto* bind_tensor = network->getInput(i);
    const char* bind_name = bind_tensor->getName();
    nvinfer1::Dims dims = bind_tensor->getDimensions();

    // Validates binding info
    if (dims.nbDims <= 0) {
      GXF_LOG_ERROR("Invalid input tensor dimensions for binding %s", bind_name);
      return ErrorCode::INVALID_ARGUMENT;
    }
    for (int j = 1; j < dims.nbDims; ++j) {
      if (dims.d[j] <= 0) {
        GXF_LOG_ERROR(
            "Input binding %s requires dynamic size on dimension No.%d which is not supported",
            bind_tensor->getName(), j);
        return ErrorCode::INVALID_ARGUMENT;
      }
    }
    if (dims.d[0] == -1) {
      // Only case with first dynamic dimension is supported and assumed to be batch size.
      // Always optimizes for 1-batch.
      dims.d[0] = 1;
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMIN, dims);
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = max_batch_size;
      if (max_batch_size <= 0) {
        dims.d[0] = 1;
      }
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMAX, dims);
    }
  }
  builderConfig->addOptimizationProfile(optimization_profile);

  // Creates TensorRT Engine Plan
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *builderConfig));
  if (!engine) {
    GXF_LOG_ERROR("Failed to build TensorRT engine from model %s.", model_file);
    return InferencerErrorCode::INVALID_ARGUMENT;
  }

  std::unique_ptr<nvinfer1::IHostMemory> model_stream(engine->serialize());
  if (!model_stream || model_stream->size() == 0 || model_stream->data() == nullptr) {
    GXF_LOG_ERROR("Fail to serialize TensorRT Engine.");
    return InferencerErrorCode::INVALID_ARGUMENT;
  }

  // Prepares return value
  const char* data = static_cast<const char*>(model_stream->data());
  m_modelEngineStream.resize(model_stream->size());
  std::copy(data, data + model_stream->size(), m_modelEngineStream.data());
  m_modelEngineStreamSize =  model_stream->size();
  return InferencerErrorCode::SUCCESS;
}

// Writes engine plan to specified file path
std::error_code SerializeEnginePlan(const std::vector<char>& plan, const std::string path) {
  // Write Plan To Disk
  std::ofstream out_stream(path.c_str(), std::ofstream::binary);
  if (!out_stream.is_open()) {
    GXF_LOG_ERROR("Failed to open engine file %s.", path.c_str());
    return InferencerErrorCode::TENSORRT_ENGINE_ERROR;
  }
  out_stream.write(plan.data(), plan.size());
  if (out_stream.bad()) {
    GXF_LOG_ERROR("Failed writing engine file %s.", path.c_str());
    return InferencerErrorCode::TENSORRT_ENGINE_ERROR;
  }
  out_stream.close();
  GXF_LOG_INFO("TensorRT engine serialized at %s", path.c_str());
  return InferencerErrorCode::SUCCESS;
}

TensorRTInferencer::TensorRTInferencer(const TensorRTInferenceParams& params)
    : m_logger(new TRTLogger())
    , m_maxBatchSize(params.maxBatchSize)
    , m_inputLayers(params.inputLayerNames)
    , m_outputLayers(params.outputLayerNames)
    , m_cudaStream(0)
    , m_inferenceEngine(nullptr) {
    if (params.inferType == TRTInferenceType::TRT_ENGINE) {
        std::ifstream trtModelFStream(params.engineFilePath, std::ios::binary);
        const bool shouldRebuild = params.force_engine_update || !trtModelFStream.good();
        const bool canRebuild = params.onnxFilePath.size() != 0;
        if (canRebuild && shouldRebuild) {
            // Deletes engine plan file if exists for forced update
            std::remove(params.engineFilePath.c_str());
            if (std::ifstream(params.engineFilePath).good()) {
                GXF_LOG_ERROR("Failed to remove engine plan file %s for forced engine update.",
                    params.engineFilePath.c_str());
            }
            GXF_LOG_INFO(
                "Rebuilding CUDA engine %s%s. "
                "Note: this process may take up to several minutes.",
                params.force_engine_update ? " (forced by config)" : "",
                params.engineFilePath.c_str());
            auto result = convertModelToEngine(params.dlaID, params.onnxFilePath.c_str(),
                params.max_workspace_size, params.buildFlags, params.maxBatchSize);
            if (result != InferencerErrorCode::SUCCESS) {
                GXF_LOG_INFO("Failed to create engine plan for model %s.",
                    params.onnxFilePath.c_str());
            }

            // Tries to serializes the plan and proceeds anyway
            if (SerializeEnginePlan(m_modelEngineStream, params.engineFilePath) !=
                InferencerErrorCode::SUCCESS) {
                GXF_LOG_INFO(
                    "Engine plan serialization failed. Proceeds with in-memory"
                    "engine plan anyway.");
            }
        } else {
            GXF_LOG_INFO("Using CUDA engine %s. ", params.engineFilePath.c_str());

            trtModelFStream.seekg(0, trtModelFStream.end);
            m_modelEngineStreamSize = trtModelFStream.tellg();
            m_modelEngineStream.resize(m_modelEngineStreamSize);
            trtModelFStream.seekg(0, trtModelFStream.beg);
            trtModelFStream.read(m_modelEngineStream.data(), m_modelEngineStreamSize);
            trtModelFStream.close();
        }

        m_inferenceRuntime.reset(nvinfer1::createInferRuntime(*(m_logger.get())));
        if (params.dlaID != -1 && params.dlaID < m_inferenceRuntime->getNbDLACores()) {
            m_inferenceRuntime->setDLACore(params.dlaID);
        }
        m_inferenceEngine = m_inferenceRuntime->deserializeCudaEngine(m_modelEngineStream.data(),
            m_modelEngineStreamSize);
        m_ownedInferenceEngine.reset(m_inferenceEngine);
        m_inferenceContext.reset(m_inferenceEngine->createExecutionContext());
        m_inferenceContext->setOptimizationProfileAsync(0, m_cudaStream);
    } else {
        if (params.engine == nullptr) {
            throw ErrorCode::INVALID_ARGUMENT;
        }
        m_inferenceEngine = params.engine;
        m_inferenceContext.reset(m_inferenceEngine->createExecutionContext());
    }

    if (m_inferenceEngine == nullptr || m_inferenceContext == nullptr) {
        throw ErrorCode::INVALID_ARGUMENT;
    }

    m_hasImplicitBatch = m_inferenceEngine->hasImplicitBatchDimension();
    m_bindingsCount    = m_inferenceEngine->getNbBindings();
    if (!m_hasImplicitBatch) {
        for (size_t i = 0; i < m_bindingsCount; i++) {
            if (m_inferenceEngine->bindingIsInput(i)) {
                nvinfer1::Dims dims_i(m_inferenceEngine->getBindingDimensions(i));
                nvinfer1::Dims4 inputDims{1, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
                m_inferenceContext->setBindingDimensions(i, inputDims);
            }
        }
    }
    std::error_code err;
    err = ParseTRTModel();
    if (err != make_error_code(ErrorCode::SUCCESS)) {
        throw err;
    }
    m_buffers.resize(m_bindingsCount);
}

// Set input layer tensor
std::error_code TensorRTInferencer::setInput(const TensorBase& trtInputBuffer,
    std::string inputLayerName) {
    if (m_modelInfo.inputLayers.find(inputLayerName) == m_modelInfo.inputLayers.end()) {
        return ErrorCode::INVALID_ARGUMENT;
    }
    LayerInfo layer        = m_modelInfo.inputLayers[inputLayerName];
    m_buffers[layer.index] = trtInputBuffer.getData();
    return ErrorCode::SUCCESS;
}

// Sets output layer tensor
std::error_code TensorRTInferencer::setOutput(TensorBase& trtOutputBuffer,
    std::string outputLayerName) {
    if (m_modelInfo.outputLayers.find(outputLayerName) == m_modelInfo.outputLayers.end()) {
        return ErrorCode::INVALID_ARGUMENT;
    }
    LayerInfo layer        = m_modelInfo.outputLayers[outputLayerName];
    m_buffers[layer.index] = trtOutputBuffer.getData();
    return ErrorCode::SUCCESS;
}

// Get the model metadata parsed based on the model file
// This would be done in initialize call itself. User can access the modelMetaData
// created using this API.
ModelMetaData TensorRTInferencer::getModelMetaData() const {
    return m_modelInfo;
}

std::error_code TensorRTInferencer::infer(size_t batchSize) {
    bool err = true;
    if (!m_hasImplicitBatch) {
        size_t bindingsCount = m_inferenceEngine->getNbBindings();
        for (size_t i = 0; i < bindingsCount; i++) {
            if (m_inferenceEngine->bindingIsInput(i)) {
                nvinfer1::Dims dims_i(m_inferenceEngine->getBindingDimensions(i));
                nvinfer1::Dims4 inputDims{static_cast<int>(batchSize), dims_i.d[1],
                    dims_i.d[2], dims_i.d[3]};
                m_inferenceContext->setBindingDimensions(i, inputDims);
            }
        }
        err = m_inferenceContext->enqueueV2(&m_buffers[0], m_cudaStream, nullptr);
    } else {
        err = m_inferenceContext->enqueue(m_maxBatchSize, &m_buffers[0], m_cudaStream, nullptr);
    }
    if (!err) {
        return InferencerErrorCode::TENSORRT_INFERENCE_ERROR;
    }
    return ErrorCode::SUCCESS;
}

// Applicable only for Native TRT
std::error_code TensorRTInferencer::setCudaStream(cudaStream_t cudaStream) {
    m_cudaStream = cudaStream;
    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::unregister(std::string layerName) {
    size_t index;
    if (m_modelInfo.outputLayers.find(layerName) != m_modelInfo.outputLayers.end()) {
        index = m_modelInfo.outputLayers[layerName].index;
    } else if (m_modelInfo.inputLayers.find(layerName) != m_modelInfo.inputLayers.end()) {
        index = m_modelInfo.inputLayers[layerName].index;
    } else {
        return ErrorCode::INVALID_ARGUMENT;
    }
    m_buffers[index] = nullptr;
    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::unregister() {
    for (size_t i = 0; i < m_buffers.size(); i++) {
        m_buffers[i] = nullptr;
    }
    return ErrorCode::SUCCESS;
}

TensorRTInferencer::~TensorRTInferencer() {
    m_buffers.clear();
}

}  // namespace inferencer
}  // namespace cvcore
