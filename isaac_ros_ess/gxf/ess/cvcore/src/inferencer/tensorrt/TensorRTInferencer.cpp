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
#include "TensorRTInferencer.h"
#include <fstream>
#include <iostream>
#include "TensorRTUtils.h"

#include "cv/inferencer/Errors.h"
#include "cv/inferencer/IInferenceBackend.h"
#include "cv/inferencer/Inferencer.h"

namespace cvcore { namespace inferencer {

namespace {
size_t getDataSize(const std::vector<int64_t> &shape, cvcore::ChannelType dataType)
{
    size_t layerShape = 1;
    for (size_t k = 0; k < shape.size(); k++)
        layerShape *= shape[k] <= 0 ? 1 : shape[k];

    return layerShape * GetChannelSize(dataType);
}
} // namespace

std::error_code TensorRTInferencer::getLayerInfo(LayerInfo &layer, std::string layerName)
{
    layer.name                          = layerName;
    layer.index                         = m_inferenceEngine->getBindingIndex(layerName.c_str());
    auto dim                            = m_inferenceEngine->getBindingDimensions(layer.index);
    nvinfer1::TensorFormat tensorFormat = m_inferenceEngine->getBindingFormat(layer.index);

    std::error_code err;
    err = getCVCoreChannelLayoutFromTensorRT(layer.layout, tensorFormat);
    if (err != cvcore::make_error_code(ErrorCode::SUCCESS))
    {
        return ErrorCode::INVALID_ARGUMENT;
    }

    for (size_t cnt = 0; cnt < dim.nbDims; cnt++)
    {
        layer.shape.push_back(dim.d[cnt]);
    }

    err = getCVCoreChannelTypeFromTensorRT(layer.dataType, m_inferenceEngine->getBindingDataType(layer.index));
    layer.layerSize = getDataSize(layer.shape, layer.dataType);
    if (err != cvcore::make_error_code(ErrorCode::SUCCESS))
    {
        return ErrorCode::INVALID_ARGUMENT;
    }

    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::ParseTRTModel()
{

    m_modelInfo.modelName    = m_inferenceEngine->getName();
    m_modelInfo.modelVersion = "";
    m_modelInfo.maxBatchSize = m_maxBatchSize;
    std::error_code err;
    for (size_t i = 0; i < m_inputLayers.size(); i++)
    {
        LayerInfo layer;
        err = getLayerInfo(layer, m_inputLayers[i]);
        if (err != cvcore::make_error_code(cvcore::ErrorCode::SUCCESS))
        {
            return err;
        }
        m_modelInfo.inputLayers[layer.name] = layer;
    }
    for (size_t i = 0; i < m_outputLayers.size(); i++)
    {
        LayerInfo layer;
        err = getLayerInfo(layer, m_outputLayers[i]);
        if (err != cvcore::make_error_code(cvcore::ErrorCode::SUCCESS))
        {
            return err;
        }
        m_modelInfo.outputLayers[layer.name] = layer;
    }

    return ErrorCode::SUCCESS;
}

TensorRTInferencer::TensorRTInferencer(const TensorRTInferenceParams &params)
    : m_logger(new TRTLogger())
    , m_maxBatchSize(params.maxBatchSize)
    , m_inputLayers(params.inputLayerNames)
    , m_outputLayers(params.outputLayerNames)
    , m_cudaStream(0)
    , m_inferenceEngine(nullptr)
{

    if (params.inferType == TRTInferenceType::TRT_ENGINE)
    {
        std::ifstream trtModelFStream(params.engineFilePath, std::ios::binary);
        std::unique_ptr<char[]> trtModelContent;
        size_t trtModelContentSize = 0;

        if (!trtModelFStream.good())
        {
            throw ErrorCode::INVALID_ARGUMENT;
        }
        else
        {
            trtModelFStream.seekg(0, trtModelFStream.end);
            trtModelContentSize = trtModelFStream.tellg();
            trtModelFStream.seekg(0, trtModelFStream.beg);
            trtModelContent.reset(new char[trtModelContentSize]);
            trtModelFStream.read(trtModelContent.get(), trtModelContentSize);
            trtModelFStream.close();
        }

        m_inferenceRuntime.reset(nvinfer1::createInferRuntime(*(m_logger.get())));
        if (params.dlaID != -1 && params.dlaID < m_inferenceRuntime->getNbDLACores())
        {
            m_inferenceRuntime->setDLACore(params.dlaID);
        }
        m_inferenceEngine = m_inferenceRuntime->deserializeCudaEngine(trtModelContent.get(), trtModelContentSize);
        m_ownedInferenceEngine.reset(m_inferenceEngine);
        m_inferenceContext.reset(m_inferenceEngine->createExecutionContext());
        m_inferenceContext->setOptimizationProfileAsync(0, m_cudaStream);
    }
    else
    {
        if (params.engine == nullptr)
        {
            throw ErrorCode::INVALID_ARGUMENT;
        }
        m_inferenceEngine = params.engine;
        m_inferenceContext.reset(m_inferenceEngine->createExecutionContext());
    }

    if (m_inferenceEngine == nullptr || m_inferenceContext == nullptr)
    {
        throw ErrorCode::INVALID_ARGUMENT;
    }

    m_hasImplicitBatch = m_inferenceEngine->hasImplicitBatchDimension();
    m_bindingsCount    = m_inferenceEngine->getNbBindings();
    if (!m_hasImplicitBatch)
    {
        for (size_t i = 0; i < m_bindingsCount; i++)
        {
            if (m_inferenceEngine->bindingIsInput(i))
            {
                nvinfer1::Dims dims_i(m_inferenceEngine->getBindingDimensions(i));
                nvinfer1::Dims4 inputDims{1, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
                m_inferenceContext->setBindingDimensions(i, inputDims);
            }
        }
    }
    std::error_code err;
    err = ParseTRTModel();
    if (err != cvcore::make_error_code(ErrorCode::SUCCESS))
    {
        throw err;
    }
    m_buffers.resize(m_bindingsCount);
}

// Set input layer tensor
std::error_code TensorRTInferencer::setInput(const cvcore::TensorBase &trtInputBuffer, std::string inputLayerName)
{
    if (m_modelInfo.inputLayers.find(inputLayerName) == m_modelInfo.inputLayers.end())
    {
        return ErrorCode::INVALID_ARGUMENT;
    }
    LayerInfo layer        = m_modelInfo.inputLayers[inputLayerName];
    m_buffers[layer.index] = trtInputBuffer.getData();
    return ErrorCode::SUCCESS;
}

// Sets output layer tensor
std::error_code TensorRTInferencer::setOutput(cvcore::TensorBase &trtOutputBuffer, std::string outputLayerName)
{
    if (m_modelInfo.outputLayers.find(outputLayerName) == m_modelInfo.outputLayers.end())
    {
        return ErrorCode::INVALID_ARGUMENT;
    }
    LayerInfo layer        = m_modelInfo.outputLayers[outputLayerName];
    m_buffers[layer.index] = trtOutputBuffer.getData();
    return ErrorCode::SUCCESS;
}

// Get the model metadata parsed based on the model file
// This would be done in initialize call itself. User can access the modelMetaData created using this API.
ModelMetaData TensorRTInferencer::getModelMetaData() const
{
    return m_modelInfo;
}

std::error_code TensorRTInferencer::infer(size_t batchSize)
{
    bool err = true;
    if (!m_hasImplicitBatch)
    {
        size_t bindingsCount = m_inferenceEngine->getNbBindings();
        for (size_t i = 0; i < bindingsCount; i++)
        {
            if (m_inferenceEngine->bindingIsInput(i))
            {
                nvinfer1::Dims dims_i(m_inferenceEngine->getBindingDimensions(i));
                nvinfer1::Dims4 inputDims{static_cast<int>(batchSize), dims_i.d[1], dims_i.d[2], dims_i.d[3]};
                m_inferenceContext->setBindingDimensions(i, inputDims);
            }
        }
        err = m_inferenceContext->enqueueV2(&m_buffers[0], m_cudaStream, nullptr);
    }
    else
    {
        err = m_inferenceContext->enqueue(m_maxBatchSize, &m_buffers[0], m_cudaStream, nullptr);
    }
    if (!err)
    {
        return InferencerErrorCode::TENSORRT_INFERENCE_ERROR;
    }
    return ErrorCode::SUCCESS;
}

// Applicable only for Native TRT
std::error_code TensorRTInferencer::setCudaStream(cudaStream_t cudaStream) // Only in TRT
{
    m_cudaStream = cudaStream;
    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::unregister(std::string layerName)
{
    size_t index;
    if (m_modelInfo.outputLayers.find(layerName) != m_modelInfo.outputLayers.end())
    {
        index = m_modelInfo.outputLayers[layerName].index;
    }
    else if (m_modelInfo.inputLayers.find(layerName) != m_modelInfo.inputLayers.end())
    {
        index = m_modelInfo.inputLayers[layerName].index;
    }
    else
    {
        return ErrorCode::INVALID_ARGUMENT;
    }
    m_buffers[index] = nullptr;
    return ErrorCode::SUCCESS;
}

std::error_code TensorRTInferencer::unregister()
{
    for (size_t i = 0; i < m_buffers.size(); i++)
    {
        m_buffers[i] = nullptr;
    }
    return ErrorCode::SUCCESS;
}

TensorRTInferencer::~TensorRTInferencer()
{
    m_buffers.clear();
}

}} // namespace cvcore::inferencer
