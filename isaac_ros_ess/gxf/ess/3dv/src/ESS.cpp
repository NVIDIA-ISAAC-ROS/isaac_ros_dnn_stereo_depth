// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cv/ess/ESS.h>

#include <cuda_runtime_api.h>

#ifdef NVBENCH_ENABLE
#include <nvbench/GPU.h>
#endif

#include <cv/core/CVError.h>
#include <cv/core/Image.h>
#include <cv/core/Memory.h>
#include <cv/inferencer/IInferenceBackend.h>
#include <cv/inferencer/Inferencer.h>

#include <cv/tensor_ops/ImageUtils.h>

namespace cvcore { namespace ess {

/* Default parameters used for the model provided*/
const ImagePreProcessingParams defaultPreProcessorParams = {
    BGR_U8,                                  /**< Input type of image.*/
    {-128, -128, -128},                      /**< Mean value */
    {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}, /**< Normalization factor */
    {0.5, 0.5, 0.5}};                        /**< Standard deviation */

const ModelInputParams defaultModelInputParams = {1,       /**< Max batch size supported */
                                                  960,     /**< Input layer width of the network */
                                                  576,     /**< Input layer height of the network */
                                                  RGB_U8}; /**< Input image type the network is trained with */

const ModelInferenceParams defaultInferenceParams = {
    "models/ess.engine",           /**< Engine file path */
    {"input_left", "input_right"}, /**< Input layer name of the model */
    {"output_left"}};              /**< Output layer name of the network */

const ESSPreProcessorParams defaultESSPreProcessorParams = {
    PreProcessType::RESIZE}; // Resize the input image to the network input dimensions

struct ESS::ESSImpl
{
    inferencer::TensorRTInferenceParams tensorrtParams;
    inferencer::InferenceBackendClient client;

    // Model includes 2 input layers and 1 output layer
    Tensor<NHWC, C1, F32> m_outputDevice;
    Tensor<NCHW, C3, F32> m_inputLeftPlanar;
    Tensor<NCHW, C3, F32> m_inputRightPlanar;

    // Preprocess and PostProcess Objects
    std::unique_ptr<ESSPreProcessor> m_preprocess;
    std::unique_ptr<ESSPostProcessor> m_postprocess;

    // Max batch Size supported
    size_t m_maxBatchSize;

    std::string m_leftInputLayerName, m_rightInputLayerName;

    size_t m_networkInputWidth, m_networkInputHeight;

    ESSImpl(const ImagePreProcessingParams &imgParams, const ModelInputParams &modelParams,
            const ModelInferenceParams &modelInferParams, const ESSPreProcessorParams &essParams)
        : m_maxBatchSize(modelParams.maxBatchSize)
    {
        if (modelInferParams.inputLayers.size() != 2 || modelInferParams.outputLayers.size() != 1 ||
            modelParams.maxBatchSize <= 0)
        {
            throw ErrorCode::INVALID_ARGUMENT;
        }

        // Initialize Preprocessor and postprocessor
        m_preprocess.reset(new ESSPreProcessor(imgParams, modelParams, essParams));
        m_postprocess.reset(new ESSPostProcessor(modelParams));

        // Initialize TRT backend
        tensorrtParams = {inferencer::TRTInferenceType::TRT_ENGINE,
                          nullptr,
                          modelInferParams.engineFilePath,
                          modelParams.maxBatchSize,
                          modelInferParams.inputLayers,
                          modelInferParams.outputLayers};

        std::error_code err =
            inferencer::InferenceBackendFactory::CreateTensorRTInferenceBackendClient(client, tensorrtParams);

        if (err != cvcore::make_error_code(cvcore::ErrorCode::SUCCESS))
        {
            throw err;
        }

        inferencer::ModelMetaData modelInfo = client->getModelMetaData();

        m_networkInputHeight  = modelParams.inputLayerHeight;
        m_networkInputWidth   = modelParams.inputLayerWidth;
        m_inputLeftPlanar     = {m_networkInputWidth, m_networkInputHeight, modelParams.maxBatchSize, false};
        m_inputRightPlanar    = {m_networkInputWidth, m_networkInputHeight, modelParams.maxBatchSize, false};
        size_t outputWidth    = modelInfo.outputLayers[modelInferParams.outputLayers[0]].shape[2];
        size_t outputHeight   = modelInfo.outputLayers[modelInferParams.outputLayers[0]].shape[1];
        m_outputDevice        = {outputWidth, outputHeight, modelParams.maxBatchSize, false};
        m_leftInputLayerName  = modelInferParams.inputLayers[0];
        m_rightInputLayerName = modelInferParams.inputLayers[1];
        CHECK_ERROR_CODE(client->setInput(m_inputLeftPlanar, modelInferParams.inputLayers[0]));
        CHECK_ERROR_CODE(client->setInput(m_inputRightPlanar, modelInferParams.inputLayers[1]));
        CHECK_ERROR_CODE(client->setOutput(m_outputDevice, modelInferParams.outputLayers[0]));
    }

    ~ESSImpl()
    {
        CHECK_ERROR_CODE(client->unregister());
        inferencer::InferenceBackendFactory::DestroyTensorRTInferenceBackendClient(client);
    }

    void execute(Tensor<NHWC, C1, F32> &output, const Tensor<NHWC, C3, U8> &inputLeft,
                 const Tensor<NHWC, C3, U8> &inputRight, cudaStream_t stream)
    {
        size_t batchSize = inputLeft.getDepth();
        if (inputLeft.isCPU() || inputRight.isCPU())
        {
            throw std::invalid_argument("ESS : Input type should be GPU buffer");
        }

        if (inputLeft.getDepth() > m_maxBatchSize || inputRight.getDepth() > m_maxBatchSize)
        {
            throw std::invalid_argument("ESS : Input batch size cannot exceed max batch size\n");
        }

        if (inputLeft.getDepth() != inputRight.getDepth() || output.getDepth() != inputLeft.getDepth())
        {
            throw std::invalid_argument("ESS : Batch size of input and output images don't match!\n");
        }
        m_preprocess->execute(m_inputLeftPlanar, m_inputRightPlanar, inputLeft, inputRight, stream);

#ifdef NVBENCH_ENABLE
        size_t inputWidth          = inputLeft.getWidth();
        size_t inputHeight         = inputLeft.getHeight();
        const std::string testName = "ESSInference_batch" + std::to_string(batchSize) + "_" +
                                     std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "x" +
                                     std::to_string(inputLeft.getChannelCount()) + "_GPU";
        nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif

        CHECK_ERROR_CODE(client->setCudaStream(stream));
        CHECK_ERROR_CODE(client->infer(batchSize));
        // PostProcess
        m_postprocess->execute(output, m_outputDevice, stream);
    }

    void execute(Tensor<NHWC, C1, F32> &output, const Tensor<NCHW, C3, F32> &inputLeft,
                 const Tensor<NCHW, C3, F32> &inputRight, cudaStream_t stream)
    {
        size_t batchSize = inputLeft.getDepth();
        if (inputLeft.isCPU() || inputRight.isCPU())
        {
            throw std::invalid_argument("ESS : Input type should be GPU buffer");
        }

        if (inputLeft.getDepth() > m_maxBatchSize || inputRight.getDepth() > m_maxBatchSize)
        {
            throw std::invalid_argument("ESS : Input batch size cannot exceed max batch size\n");
        }

        if (inputLeft.getDepth() != inputRight.getDepth() || output.getDepth() != inputLeft.getDepth())
        {
            throw std::invalid_argument("ESS : Batch size of input and output images don't match!\n");
        }

        if (inputLeft.getWidth() != m_networkInputWidth || inputLeft.getHeight() != m_networkInputHeight)
        {
            throw std::invalid_argument("ESS : Left preprocessed input does not match network input dimensions!\n");
        }

        if (inputRight.getWidth() != m_networkInputWidth || inputRight.getHeight() != m_networkInputHeight)
        {
            throw std::invalid_argument("ESS : Right preprocessed input does not match network input dimensions!\n");
        }
#ifdef NVBENCH_ENABLE
        size_t inputWidth          = inputLeft.getWidth();
        size_t inputHeight         = inputLeft.getHeight();
        const std::string testName = "ESSInference_batch" + std::to_string(batchSize) + "_" +
                                     std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + "x" +
                                     std::to_string(inputLeft.getChannelCount()) + "_GPU";
        nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
        // inference
        CHECK_ERROR_CODE(client->setInput(inputLeft, m_leftInputLayerName));
        CHECK_ERROR_CODE(client->setInput(inputRight, m_rightInputLayerName));

        CHECK_ERROR_CODE(client->setCudaStream(stream));
        CHECK_ERROR_CODE(client->infer(batchSize));
        // PostProcess
        m_postprocess->execute(output, m_outputDevice, stream);
    }
};

ESS::ESS(const ImagePreProcessingParams &imgParams, const ModelInputParams &modelParams,
         const ModelInferenceParams &modelInferParams, const ESSPreProcessorParams &essParams)
    : m_pImpl(new ESSImpl(imgParams, modelParams, modelInferParams, essParams))
{
}

ESS::~ESS() {}

void ESS::execute(Tensor<NHWC, C1, F32> &output, const Tensor<NCHW, C3, F32> &inputLeft,
                  const Tensor<NCHW, C3, F32> &inputRight, cudaStream_t stream)
{
    m_pImpl->execute(output, inputLeft, inputRight, stream);
}

void ESS::execute(Tensor<NHWC, C1, F32> &output, const Tensor<NHWC, C3, U8> &inputLeft,
                  const Tensor<NHWC, C3, U8> &inputRight, cudaStream_t stream)
{
    m_pImpl->execute(output, inputLeft, inputRight, stream);
}
}} // namespace cvcore::ess
