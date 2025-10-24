// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/ess/inference/ESS.h"

#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "extensions/tensorops/core/CVError.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"
#include "gems/dnn_inferencer/inferencer/IInferenceBackend.h"
#include "gems/dnn_inferencer/inferencer/Inferencer.h"

namespace nvidia {
namespace isaac {
namespace ess {
using ImagePreProcessingParams = cvcore::tensor_ops::ImagePreProcessingParams;
using TensorRTInferenceParams = cvcore::inferencer::TensorRTInferenceParams;
using InferenceBackendClient = cvcore::inferencer::InferenceBackendClient;
using InferenceBackendFactory = cvcore::inferencer::InferenceBackendFactory;
using TRTInferenceType = cvcore::inferencer::TRTInferenceType;
using ErrorCode = cvcore::tensor_ops::ErrorCode;

/* Default parameters used for the model provided*/
const ImagePreProcessingParams defaultPreProcessorParams = {
    ImageType::BGR_U8,   /**< Input type of image.*/
    {-128, -128, -128},                      /**< Mean value */
    {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}, /**< Normalization factor */
    {0.5, 0.5, 0.5}};                        /**< Standard deviation */

const ModelInputParams defaultModelInputParams = {1,       /**< Max batch size supported */
    ImageType::RGB_U8}; /**< Input image type the network is trained with */

const ModelInferenceParams defaultInferenceParams = {
    "models/ess.engine",           /**< Engine file path */
    {"input_left", "input_right"}, /**< Input layer name of the model */
    {"output_left", "output_conf"}};              /**< Output layer name of the network */

const ESSPreProcessorParams defaultESSPreProcessorParams = {
    PreProcessType::RESIZE};  // Resize the input image to the network input dimensions

const int InferencerVersion = NV_TENSORRT_VERSION;

struct ESS::ESSImpl {
    TensorRTInferenceParams tensorrtParams;
    InferenceBackendClient client;

    // Model includes 2 input layers and 1 output layer
    Tensor_NHWC_C1_F32 m_outputDevice;
    Tensor_NHWC_C1_F32 m_confMap;
    Tensor_NCHW_C3_F32 m_inputLeftPlanar;
    Tensor_NCHW_C3_F32 m_inputRightPlanar;

    // Preprocess Objects
    std::unique_ptr<ESSPreProcessor> m_preprocess;

    // Max batch Size supported
    size_t m_maxBatchSize;

    std::string m_leftInputLayerName, m_rightInputLayerName;
    std::string m_outputName;
    std::string m_confMapName;

    size_t m_networkInputWidth, m_networkInputHeight;
    size_t m_networkOutputWidth, m_networkOutputHeight;

    ESSImpl(const ImagePreProcessingParams & imgParams,
        const ModelInputParams & modelInputParams,
        const ModelBuildParams & modelBuildParams,
        const ModelInferenceParams & modelInferParams,
        const ESSPreProcessorParams & essParams)
        : m_maxBatchSize(modelInputParams.maxBatchSize) {
        if (modelInferParams.inputLayers.size() != 2 || modelInferParams.outputLayers.size() != 2 ||
            modelInputParams.maxBatchSize <= 0) {
            throw ErrorCode::INVALID_ARGUMENT;
        }

        cvcore::inferencer::OnnxModelBuildFlag buildFlags =
            cvcore::inferencer::OnnxModelBuildFlag::NONE;
        if (modelBuildParams.enable_fp16) {
            buildFlags = cvcore::inferencer::OnnxModelBuildFlag::kFP16;
        }
        // Initialize TRT backend
        tensorrtParams = {TRTInferenceType::TRT_ENGINE,
                          nullptr,
                          modelBuildParams.onnx_file_path,
                          modelInferParams.engineFilePath,
                          modelBuildParams.force_engine_update,
                          buildFlags,
                          modelBuildParams.max_workspace_size,
                          modelInputParams.maxBatchSize,
                          modelInferParams.inputLayers,
                          modelInferParams.outputLayers,
                          modelBuildParams.dla_core};

        std::error_code err =
            InferenceBackendFactory::CreateTensorRTInferenceBackendClient(client, tensorrtParams);

        if (err != cvcore::tensor_ops::make_error_code(ErrorCode::SUCCESS)) {
            throw err;
        }

        cvcore::inferencer::ModelMetaData modelInfo = client->getModelMetaData();
        if (modelInfo.maxBatchSize != modelInputParams.maxBatchSize) {
            throw ErrorCode::INVALID_ARGUMENT;
        }
        m_leftInputLayerName  = modelInferParams.inputLayers[0];
        m_rightInputLayerName = modelInferParams.inputLayers[1];
        m_outputName = modelInferParams.outputLayers[0];
        m_confMapName = modelInferParams.outputLayers[1];

        m_networkInputHeight  = modelInfo.inputLayers[m_leftInputLayerName].shape[2];
        m_networkInputWidth   = modelInfo.inputLayers[m_leftInputLayerName].shape[3];
        m_inputLeftPlanar     = {m_networkInputWidth, m_networkInputHeight,
            modelInputParams.maxBatchSize, false};
        m_inputRightPlanar    = {m_networkInputWidth, m_networkInputHeight,
            modelInputParams.maxBatchSize, false};

        m_networkOutputHeight = modelInfo.outputLayers[modelInferParams.outputLayers[0]].shape[1];
        m_networkOutputWidth  = modelInfo.outputLayers[modelInferParams.outputLayers[0]].shape[2];
        m_outputDevice = {m_networkOutputWidth, m_networkOutputHeight,
            modelInputParams.maxBatchSize, false};
        m_confMap = {m_networkOutputWidth, m_networkOutputHeight,
            modelInputParams.maxBatchSize, false};

        CHECK_ERROR_CODE(client->setInput(m_inputLeftPlanar, modelInferParams.inputLayers[0]));
        CHECK_ERROR_CODE(client->setInput(m_inputRightPlanar, modelInferParams.inputLayers[1]));

        // Initialize Preprocessor
        m_preprocess.reset(new ESSPreProcessor(imgParams, modelInputParams,
            m_networkInputWidth, m_networkInputHeight, essParams));
    }

    ~ESSImpl() {
        CHECK_ERROR_CODE(client->unregister());
        InferenceBackendFactory::DestroyTensorRTInferenceBackendClient(client);
    }

    size_t getModelOutputHeight() {
        return m_networkOutputHeight;
    }

    size_t getModelOutputWidth() {
        return m_networkOutputWidth;
    }

    void execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
                 cudaStream_t stream) {
        size_t batchSize = inputLeft.getDepth();
        if (inputLeft.isCPU() || inputRight.isCPU()) {
            throw std::invalid_argument("ESS : Input type should be GPU buffer");
        }

        if (inputLeft.getDepth() > m_maxBatchSize || inputRight.getDepth() > m_maxBatchSize) {
            throw std::invalid_argument("ESS : Input batch size cannot exceed max batch size\n");
        }

        if (inputLeft.getDepth() != inputRight.getDepth() ||
            output.getDepth() != inputLeft.getDepth()) {
            throw std::invalid_argument("ESS : Batch size of input and output images don't"
                "match!\n");
        }
        m_preprocess->execute(m_inputLeftPlanar, m_inputRightPlanar, inputLeft,
            inputRight, stream);

        CHECK_ERROR_CODE(client->setOutput(output,  m_outputName));
        CHECK_ERROR_CODE(client->setOutput(confMap, m_confMapName));
        CHECK_ERROR_CODE(client->setCudaStream(stream));
        CHECK_ERROR_CODE(client->infer(batchSize));
        CHECK_ERROR(cudaStreamSynchronize(stream));
    }

    void execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                 const Tensor_NCHW_C3_F32 & inputLeft, const Tensor_NCHW_C3_F32 & inputRight,
                 cudaStream_t stream) {
        size_t batchSize = inputLeft.getDepth();
        if (inputLeft.isCPU() || inputRight.isCPU()) {
            throw std::invalid_argument("ESS : Input type should be GPU buffer");
        }

        if (inputLeft.getDepth() > m_maxBatchSize || inputRight.getDepth() > m_maxBatchSize) {
            throw std::invalid_argument("ESS : Input batch size cannot exceed max batch size\n");
        }

        if (inputLeft.getDepth() != inputRight.getDepth() ||
            output.getDepth() != inputLeft.getDepth()) {
            throw std::invalid_argument("ESS : Batch size of input and output images don't"
                "match!\n");
        }

        if (inputLeft.getWidth() != m_networkInputWidth ||
            inputLeft.getHeight() != m_networkInputHeight) {
            throw std::invalid_argument("ESS : Left preprocessed input does not match network"
                "input dimensions!\n");
        }

        if (inputRight.getWidth() != m_networkInputWidth ||
            inputRight.getHeight() != m_networkInputHeight) {
            throw std::invalid_argument("ESS : Right preprocessed input does not match network"
                "input dimensions!\n");
        }

        // inference
        CHECK_ERROR_CODE(client->setInput(inputLeft, m_leftInputLayerName));
        CHECK_ERROR_CODE(client->setInput(inputRight, m_rightInputLayerName));
        CHECK_ERROR_CODE(client->setOutput(output,  m_outputName));
        CHECK_ERROR_CODE(client->setOutput(confMap, m_confMapName));
        CHECK_ERROR_CODE(client->setCudaStream(stream));
        CHECK_ERROR_CODE(client->infer(batchSize));
        CHECK_ERROR(cudaStreamSynchronize(stream));
    }
};

ESS::ESS(const ImagePreProcessingParams & imgParams,
         const ModelInputParams & modelInputParams,
         const ModelBuildParams & modelBuildParams,
         const ModelInferenceParams & modelInferParams,
         const ESSPreProcessorParams & essParams)
    : m_pImpl(new ESSImpl(imgParams, modelInputParams,
              modelBuildParams, modelInferParams, essParams)) {}

ESS::~ESS() {}

void ESS::execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                  const Tensor_NCHW_C3_F32 & inputLeft, const Tensor_NCHW_C3_F32 & inputRight,
                  cudaStream_t stream) {
    m_pImpl->execute(output, confMap, inputLeft, inputRight, stream);
}

void ESS::execute(Tensor_NHWC_C1_F32 & output, Tensor_NHWC_C1_F32 & confMap,
                  const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
                  cudaStream_t stream) {
    m_pImpl->execute(output, confMap, inputLeft, inputRight, stream);
}

size_t ESS::getModelOutputHeight() {
    return m_pImpl->getModelOutputHeight();
}

size_t ESS::getModelOutputWidth() {
    return m_pImpl->getModelOutputWidth();
}

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia
