// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cv/trtbackend/TRTBackend.h"

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvUtils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cvcore {

namespace {


void WriteSerializedEngineToFile(const char *data, size_t engineSize, std::string &outputFile)
{
    std::ofstream outFile(outputFile.c_str(), std::ios::binary);
    if (!outFile.is_open())
    {
        throw std::runtime_error("Cannot open file to write serialized Engine. Permissions? ");
    }
    else
    {
        outFile.write(data, engineSize);
        outFile.close();
    }
}

nvinfer1::Dims4 GetTRTBlobDimension(int batch, int channels, int height, int width, TRTBackendBlobLayout layout)
{
    nvinfer1::Dims4 dims;
    switch (layout)
    {
    case TRTBackendBlobLayout::PLANAR:
    {
        dims = {batch, channels, height, width};
        break;
    }
    case TRTBackendBlobLayout::INTERLEAVED:
    {
        dims = {batch, height, width, channels};
        break;
    }
    default:
    {
        throw std::runtime_error("Only PLANAR and INTERLEAVED types allowed");
    }
    }
    return dims;
}

nvinfer1::Dims3 GetTRTBlobDimension(int channels, int height, int width, TRTBackendBlobLayout layout)
{
    nvinfer1::Dims3 dims;
    switch (layout)
    {
    case TRTBackendBlobLayout::PLANAR:
    {
        dims = {channels, height, width};
        break;
    }
    case TRTBackendBlobLayout::INTERLEAVED:
    {
        dims = {height, width, channels};
        break;
    }
    default:
    {
        throw std::runtime_error("Only PLANAR and INTERLEAVED types allowed");
    }
    }
    return dims;
}

bool SetupProfile(nvinfer1::IOptimizationProfile *profile, nvinfer1::INetworkDefinition *network,
                  TRTBackendParams &params)
{
    // This shouldn't be hard-coded rather should be set by the user.
    int kMINBatchSize = 1;
    int kMAXBatchSize = 32;

    if (kMAXBatchSize < params.batchSize)
    {
        throw std::runtime_error("Max batch size is hard-coded to 32.");
    }

    bool hasDynamicShape = false;
    for (int i = 0; i < network->getNbInputs(); i++)
    {
        auto input                = network->getInput(i);
        nvinfer1::Dims dims       = input->getDimensions();
        const bool isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; });
        if (isDynamicInput)
        {
            hasDynamicShape = true;
            auto it = std::find(params.inputLayers.begin(), params.inputLayers.end(), std::string(input->getName()));
            if (it == params.inputLayers.end())
            {
                throw std::runtime_error("Undefined dynamic input shape");
            }
            int pos       = it - params.inputLayers.begin();
            auto inputDim = params.inputDims[pos];
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                                   GetTRTBlobDimension(kMINBatchSize, inputDim.channels, inputDim.height,
                                                       inputDim.width, params.inputLayout));
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                                   GetTRTBlobDimension(params.batchSize, inputDim.channels, inputDim.height,
                                                       inputDim.width, params.inputLayout));
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                                   GetTRTBlobDimension(kMAXBatchSize, inputDim.channels, inputDim.height,
                                                       inputDim.width, params.inputLayout));
        }
    }
    return hasDynamicShape;
}

nvinfer1::DataType GetTRTDataType(TRTBackendPrecision precision)
{
    nvinfer1::DataType dataType;
    switch (precision)
    {
    case TRTBackendPrecision::INT8:
    {
        dataType = nvinfer1::DataType::kINT8;
        break;
    }
    case TRTBackendPrecision::FP16:
    {
        dataType = nvinfer1::DataType::kHALF;
        break;
    }
    case TRTBackendPrecision::FP32:
    {
        dataType = nvinfer1::DataType::kFLOAT;
        break;
    }
    default:
    {
        dataType = nvinfer1::DataType::kFLOAT;
        break;
    }
    }
    return dataType;
}

nvuffparser::UffInputOrder GetTRTInputOrder(TRTBackendBlobLayout layout)
{
    nvuffparser::UffInputOrder order;
    switch (layout)
    {
    case TRTBackendBlobLayout::PLANAR:
    {
        order = nvuffparser::UffInputOrder::kNCHW;
        break;
    }
    case TRTBackendBlobLayout::INTERLEAVED:
    {
        order = nvuffparser::UffInputOrder::kNHWC;
        break;
    }
    default:
    {
        throw std::runtime_error("Only PLANAR and INTERLEAVED types allowed");
    }
    }
    return order;
}

} // anonymous namespace

class TRTLogger : public nvinfer1::ILogger
{

public:
    nvinfer1::ILogger &getLogger()
    {
        return *this;
    }

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        {
            std::cout << msg << std::endl;
            break;
        }
        case nvinfer1::ILogger::Severity::kERROR:
        {
            std::cout << msg << std::endl;
            break;
        }
        case nvinfer1::ILogger::Severity::kWARNING:
        {
            std::cout << msg << std::endl;
            break;
        }
        case nvinfer1::ILogger::Severity::kINFO:
        {
            std::cout << msg << std::endl;
            break;
        }
        default:
        {
            std::cout << msg << std::endl;
            break;
        }
        }
    }
};

struct TRTImpl
{
    TRTImpl()
        : m_logger(new TRTLogger())
        , m_TRTRuntime(nullptr, [](nvinfer1::IRuntime *runtime) { runtime->destroy(); })
        , m_inferenceEngine(nullptr)
        , m_ownedInferenceEngine(nullptr, [](nvinfer1::ICudaEngine *eng) { eng->destroy(); })
        , m_inferContext(nullptr, [](nvinfer1::IExecutionContext *ectx) { ectx->destroy(); })
        , m_cudaStream(0)
    {
    }

    std::unique_ptr<TRTLogger> m_logger;
    std::unique_ptr<nvinfer1::IRuntime, void (*)(nvinfer1::IRuntime *)> m_TRTRuntime;
    nvinfer1::ICudaEngine *m_inferenceEngine;
    std::unique_ptr<nvinfer1::ICudaEngine, void (*)(nvinfer1::ICudaEngine *)> m_ownedInferenceEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, void (*)(nvinfer1::IExecutionContext *)> m_inferContext;

    cudaStream_t m_cudaStream;
    int m_bindingsCount             = 0;
    int m_batchSize                 = 1;
    bool m_explicitBatch            = false;
    TRTBackendPrecision m_precision = TRTBackendPrecision::FP32;
    std::unordered_map<std::string, int> m_blobMap;

    void loadNetWorkFromFile(const char *modelFilePath);
    void loadNetWorkFromUff(TRTBackendParams &params);
    void loadNetWorkFromOnnx(TRTBackendParams &params);
    void loadFromMemoryPointer(void *engine);
    // find the input/output bindings
    void setupIO(int batchSize);
};

void TRTImpl::loadNetWorkFromFile(const char *modelFilePath)
{
    // Initialize TRT engine and deserialize it from file
    std::ifstream trtModelFStream(modelFilePath, std::ios::binary);
    std::unique_ptr<char[]> trtModelContent;
    size_t trtModelContentSize = 0;
    if (!trtModelFStream.good())
    {
        std::cerr << "Model File: " << modelFilePath << std::endl;
        throw std::runtime_error("TensorRT: Model file not found.");
    }
    else
    {
        trtModelFStream.seekg(0, trtModelFStream.end);
        trtModelContentSize = trtModelFStream.tellg();
        trtModelFStream.seekg(0, trtModelFStream.beg);
        trtModelContent.reset(new char[trtModelContentSize]);
        trtModelFStream.read(trtModelContent.get(), trtModelContentSize);
        trtModelFStream.close();
        std::cout << "Deserializing engine from: " << modelFilePath;
    }
    m_TRTRuntime.reset(nvinfer1::createInferRuntime(*(m_logger.get())));
    m_inferenceEngine = dynamic_cast<nvinfer1::ICudaEngine *>(
        m_TRTRuntime->deserializeCudaEngine(trtModelContent.get(), trtModelContentSize, nullptr));
    m_ownedInferenceEngine.reset(m_inferenceEngine);
    m_inferContext.reset(m_inferenceEngine->createExecutionContext());
    m_inferContext->setOptimizationProfile(0);
}

void TRTImpl::loadNetWorkFromOnnx(TRTBackendParams &params)
{
    if (!params.explicitBatch)
    {
        std::cerr << "ONNX model only supports explicit batch size";
    }
    std::ifstream f(params.enginePath);
    if (f.good())
    {
        loadNetWorkFromFile(params.enginePath.c_str());
        return;
    }
    auto builder   = nvinfer1::createInferBuilder(*(m_logger.get()));
    auto config    = builder->createBuilderConfig();
    auto batchFlag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network   = builder->createNetworkV2(batchFlag);
    auto parser    = nvonnxparser::createParser(*network, *(m_logger.get()));

    // Force FP32
    if (!builder->platformHasFastFp16())
    {
        m_precision = TRTBackendPrecision::FP32;
    }

    // Configuration
    builder->setMaxBatchSize(params.batchSize);
    if (!parser->parseFromFile(params.weightsPath.c_str(), 0))
    {
        std::cerr << "Fail to parse";
    }
    config->setMaxWorkspaceSize(1 << 30);
    if (m_precision == TRTBackendPrecision::FP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto profile = builder->createOptimizationProfile();
    if (SetupProfile(profile, network, params))
    {
        config->addOptimizationProfile(profile);
    }

    // Build the engine
    m_inferenceEngine = builder->buildEngineWithConfig(*network, *config);
    if (m_inferenceEngine == nullptr)
    {
        throw std::runtime_error("TensorRT: unable to create engine");
    }

    m_ownedInferenceEngine.reset(m_inferenceEngine);
    m_inferContext.reset(m_inferenceEngine->createExecutionContext());

    network->destroy();
    builder->destroy();
    config->destroy();

    auto serializedEngine = m_inferenceEngine->serialize();
    WriteSerializedEngineToFile(static_cast<const char *>(serializedEngine->data()), serializedEngine->size(),
                                params.enginePath);
}

void TRTImpl::loadNetWorkFromUff(TRTBackendParams &params)
{
    auto builder   = nvinfer1::createInferBuilder(*(m_logger.get()));
    auto config    = builder->createBuilderConfig();
    auto batchFlag = params.explicitBatch
                         ? 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
                         : 0U;
    auto network = builder->createNetworkV2(batchFlag);
    auto parser  = nvuffparser::createUffParser();

    std::ifstream f(params.enginePath);
    if (f.good())
    {
        loadNetWorkFromFile(params.enginePath.c_str());
        return;
    }
    nvinfer1::DataType dataType = GetTRTDataType(params.precision);

    // Force FP32
    if (!builder->platformHasFastFp16())
    {
        dataType    = nvinfer1::DataType::kFLOAT;
        m_precision = TRTBackendPrecision::FP32;
    }

    // Register uff input
    for (int i = 0; i < params.inputLayers.size(); i++)
    {
        if (params.explicitBatch)
        {
            parser->registerInput(
                params.inputLayers[i].c_str(),
                GetTRTBlobDimension(params.batchSize, params.inputDims[i].channels, params.inputDims[i].height,
                                    params.inputDims[i].width, params.inputLayout),
                GetTRTInputOrder(params.inputLayout));
        }
        else
        {
            parser->registerInput(params.inputLayers[i].c_str(),
                                  GetTRTBlobDimension(params.inputDims[i].channels, params.inputDims[i].height,
                                                      params.inputDims[i].width, params.inputLayout),
                                  GetTRTInputOrder(params.inputLayout));
        }
    }

    // Register uff output
    for (int i = 0; i < params.outputLayers.size(); i++)
    {
        parser->registerOutput(params.outputLayers[i].c_str());
    }

    // Parse uff model
    if (!parser->parse(params.weightsPath.c_str(), *network, dataType))
    {
        std::cerr << "Fail to parse";
    }

    // Configuration
    if (params.explicitBatch)
    {
        auto profile = builder->createOptimizationProfile();
        if (SetupProfile(profile, network, params))
        {
            config->addOptimizationProfile(profile);
        }
    }
    else
    {
        builder->setMaxBatchSize(params.batchSize);
    }

    config->setMaxWorkspaceSize(1 << 30);
    if (m_precision == TRTBackendPrecision::FP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Build the engine
    m_inferenceEngine = builder->buildEngineWithConfig(*network, *config);
    if (m_inferenceEngine == nullptr)
    {
        throw std::runtime_error("TensorRT: unable to create engine");
    }

    m_ownedInferenceEngine.reset(m_inferenceEngine);
    m_inferContext.reset(m_inferenceEngine->createExecutionContext());

    network->destroy();
    builder->destroy();
    config->destroy();

    auto serializedEngine = m_inferenceEngine->serialize();
    WriteSerializedEngineToFile(static_cast<const char *>(serializedEngine->data()), serializedEngine->size(),
                                params.enginePath);
}

void TRTImpl::loadFromMemoryPointer(void *engine)
{
    m_inferenceEngine = static_cast<nvinfer1::ICudaEngine *>(engine);
    m_inferContext.reset(m_inferenceEngine->createExecutionContext());
}

void TRTImpl::setupIO(int batchSize)
{
    // @TODO: use getBindingDimensions to avoid re-setting the IO.
    m_bindingsCount = m_inferenceEngine->getNbBindings();
    for (int i = 0; i < m_bindingsCount; i++)
    {
        m_blobMap[std::string(m_inferenceEngine->getBindingName(i))] = i;
        if (m_inferenceEngine->bindingIsInput(i))
        {
            nvinfer1::Dims dims_i(m_inferenceEngine->getBindingDimensions(i));
            nvinfer1::Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
            m_inferContext->setBindingDimensions(i, inputDims);
        }
    }
}

TRTBackend::TRTBackend(const char *modelFilePath, TRTBackendPrecision precision, int batchSize, bool explicitBatch)
    : m_pImpl(new TRTImpl())
{
    m_pImpl->m_precision     = precision;
    m_pImpl->m_batchSize     = batchSize;
    m_pImpl->m_explicitBatch = explicitBatch;
    m_pImpl->loadNetWorkFromFile(modelFilePath);
    m_pImpl->setupIO(m_pImpl->m_batchSize);
}

TRTBackend::TRTBackend(TRTBackendParams &inputParams)
    : m_pImpl(new TRTImpl())
{
    m_pImpl->m_precision     = inputParams.precision;
    m_pImpl->m_batchSize     = inputParams.batchSize;
    m_pImpl->m_explicitBatch = inputParams.explicitBatch;
    switch (inputParams.modelType)
    {
    case ModelType::ONNX:
    {
        m_pImpl->loadNetWorkFromOnnx(inputParams);
        break;
    }
    case ModelType::UFF:
    {
        m_pImpl->loadNetWorkFromUff(inputParams);
        break;
    }
    case ModelType::TRT_ENGINE:
    {
        m_pImpl->loadNetWorkFromFile(inputParams.enginePath.c_str());
        break;
    }
    case ModelType::TRT_ENGINE_IN_MEMORY:
    {
        m_pImpl->loadFromMemoryPointer(inputParams.trtEngineInMemory);
        break;
    }
    default:
    {
        throw std::runtime_error(
            "Only Model types ONNX, UFF, TensorRT "
            "serialized engines and a pointer to deserialized "
            "ICudaEngine are supported\n");
    }
    }
    m_pImpl->setupIO(m_pImpl->m_batchSize);
}

TRTBackend::~TRTBackend() {}

void TRTBackend::infer(void **buffer)
{
    bool success = true;
    if (!m_pImpl->m_inferenceEngine->hasImplicitBatchDimension())
    {
        m_pImpl->m_inferContext->enqueueV2(buffer, m_pImpl->m_cudaStream, nullptr);
    }
    else
    {
        m_pImpl->m_inferContext->enqueue(m_pImpl->m_batchSize, buffer, m_pImpl->m_cudaStream, nullptr);
    }

    if (!success)
    {
        throw std::runtime_error("TensorRT: Inference failed");
    }
}

void TRTBackend::infer(void **buffer, int batchSize, cudaStream_t stream)
{
    //@TODO: fix kMin, kOpt, kMax batch size in SetupProfile() call and then add a check here.
    m_pImpl->setupIO(batchSize);

    bool success = true;
    if (!m_pImpl->m_inferenceEngine->hasImplicitBatchDimension())
    {
        m_pImpl->m_inferContext->enqueueV2(buffer, stream, nullptr);
    }
    else
    {
        m_pImpl->m_inferContext->enqueue(batchSize, buffer, stream, nullptr);
    }

    if (!success)
    {
        throw std::runtime_error("TensorRT: Inference failed");
    }
}

cudaStream_t TRTBackend::getCUDAStream() const
{
    return m_pImpl->m_cudaStream;
}

void TRTBackend::setCUDAStream(cudaStream_t stream)
{
    m_pImpl->m_cudaStream = stream;
}

int TRTBackend::getBlobCount() const
{
    return m_pImpl->m_bindingsCount;
}

TRTBackendBlobSize TRTBackend::getTRTBackendBlobSize(int blobIndex) const
{
    if (blobIndex >= m_pImpl->m_bindingsCount)
    {
        throw std::runtime_error("blobIndex out of range");
    }
    auto dim = m_pImpl->m_inferenceEngine->getBindingDimensions(blobIndex);
    TRTBackendBlobSize blobSize;
    if (dim.nbDims == 2)
    {
        blobSize = {1, dim.d[0], dim.d[1]};
    }
    else if (dim.nbDims == 3)
    {
        blobSize = {dim.d[0], dim.d[1], dim.d[2]};
    }
    else if (dim.nbDims == 4)
    {
        blobSize = {dim.d[1], dim.d[2], dim.d[3]};
    }
    else
    {
        throw std::runtime_error("Unknown TensorRT binding dimension!");
    }
    return blobSize;
}

int TRTBackend::getBlobLinearSize(int blobIndex) const
{
    const TRTBackendBlobSize shape = getTRTBackendBlobSize(blobIndex);
    nvinfer1::Dims3 dims_val{shape.channels, shape.height, shape.width};
    int blobSize = 1;
    for (int i = 0; i < 3; i++)
    {
        blobSize *= dims_val.d[i] <= 0 ? 1 : dims_val.d[i];
    }
    return blobSize;
}

int TRTBackend::getBlobIndex(const char *blobName) const
{
    auto blobItr = m_pImpl->m_blobMap.find(std::string(blobName));
    if (blobItr == m_pImpl->m_blobMap.end())
    {
        throw std::runtime_error("blobName not found");
    }
    return blobItr->second;
}

bool TRTBackend::bindingIsInput(const int index) const
{
    return m_pImpl->m_inferenceEngine->bindingIsInput(index);
}

} // namespace cvcore
