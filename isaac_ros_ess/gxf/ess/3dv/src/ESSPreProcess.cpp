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

#include <stdexcept>

#ifdef NVBENCH_ENABLE
#include <nvbench/GPU.h>
#endif

#include <cv/core/CVError.h>
#include <cv/core/Memory.h>
#include <cv/tensor_ops/BBoxUtils.h>
#include <cv/tensor_ops/ImageUtils.h>

namespace cvcore { namespace ess {

struct ESSPreProcessor::ESSPreProcessorImpl
{
    size_t m_maxBatchSize;
    size_t m_outputWidth;
    size_t m_outputHeight;
    PreProcessType m_processType;
    ImagePreProcessingParams m_preProcessorParams;
    Tensor<NHWC, C3, U8> m_resizedDeviceLeftInput;
    Tensor<NHWC, C3, U8> m_resizedDeviceRightInput;
    Tensor<NHWC, C3, F32> m_normalizedDeviceLeftInput;
    Tensor<NHWC, C3, F32> m_normalizedDeviceRightInput;
    bool m_swapRB;

    ESSPreProcessorImpl(const ImagePreProcessingParams &imgParams, const ModelInputParams &modelParams,
                        const ESSPreProcessorParams &essParams)
        : m_maxBatchSize(modelParams.maxBatchSize)
        , m_outputHeight(modelParams.inputLayerHeight)
        , m_outputWidth(modelParams.inputLayerWidth)
        , m_processType(essParams.preProcessType)
        , m_preProcessorParams(imgParams)
    {
        if (imgParams.imgType != BGR_U8 && imgParams.imgType != RGB_U8)
        {
            throw std::invalid_argument("ESSPreProcessor : Only image types RGB_U8/BGR_U8 are supported\n");
        }
        m_resizedDeviceLeftInput = {modelParams.inputLayerWidth, modelParams.inputLayerHeight, modelParams.maxBatchSize,
                                    false};
        m_resizedDeviceRightInput    = {modelParams.inputLayerWidth, modelParams.inputLayerHeight,
                                     modelParams.maxBatchSize, false};
        m_normalizedDeviceLeftInput  = {modelParams.inputLayerWidth, modelParams.inputLayerHeight,
                                       modelParams.maxBatchSize, false};
        m_normalizedDeviceRightInput = {modelParams.inputLayerWidth, modelParams.inputLayerHeight,
                                        modelParams.maxBatchSize, false};
        m_swapRB                     = imgParams.imgType != modelParams.modelInputType;
    }

    void process(Tensor<NCHW, C3, F32> &outputLeft, Tensor<NCHW, C3, F32> &outputRight,
                 const Tensor<NHWC, C3, U8> &inputLeft, const Tensor<NHWC, C3, U8> &inputRight, cudaStream_t stream)
    {
        if (inputLeft.isCPU() || inputRight.isCPU() || outputLeft.isCPU() || outputRight.isCPU())
        {
            throw std::invalid_argument("ESSPreProcessor : Input/Output Tensor must be GPU Tensor.");
        }

        if (outputLeft.getWidth() != m_outputWidth || outputLeft.getHeight() != m_outputHeight ||
            outputRight.getWidth() != m_outputWidth || outputRight.getHeight() != m_outputHeight)
        {
            throw std::invalid_argument(
                "ESSPreProcessor : Output Tensor dimension does not match network input requirement");
        }

        if (inputLeft.getWidth() != inputRight.getWidth() || inputLeft.getHeight() != inputRight.getHeight())
        {
            throw std::invalid_argument("ESSPreProcessor : Input tensor dimensions don't match");
        }

        if (outputLeft.getDepth() != inputLeft.getDepth() || outputRight.getDepth() != inputRight.getDepth() ||
            inputLeft.getDepth() != inputRight.getDepth())
        {
            throw std::invalid_argument("ESSPreProcessor : Input/Output Tensor batchsize mismatch.");
        }

        if (outputLeft.getDepth() > m_maxBatchSize)
        {
            throw std::invalid_argument("ESSPreProcessor : Input/Output batchsize exceeds max batch size.");
        }

        const size_t batchSize   = inputLeft.getDepth();
        const size_t inputWidth  = inputLeft.getWidth();
        const size_t inputHeight = inputLeft.getHeight();

        if (m_processType == PreProcessType::RESIZE)
        {
            tensor_ops::Resize(m_resizedDeviceLeftInput, inputLeft, stream);
            tensor_ops::Resize(m_resizedDeviceRightInput, inputRight, stream);
        }
        else
        {
            const float centerX = inputWidth / 2;
            const float centerY = inputHeight / 2;
            const float offsetX = m_outputWidth / 2;
            const float offsetY = m_outputHeight / 2;
            BBox srcCrop, dstCrop;
            dstCrop      = {0, 0, static_cast<int>(m_outputWidth - 1), static_cast<int>(m_outputHeight - 1)};
            srcCrop.xmin = std::max(0, static_cast<int>(centerX - offsetX));
            srcCrop.ymin = std::max(0, static_cast<int>(centerY - offsetY));
            srcCrop.xmax = std::min(static_cast<int>(m_outputWidth - 1), static_cast<int>(centerX + offsetX));
            srcCrop.ymax = std::min(static_cast<int>(m_outputHeight - 1), static_cast<int>(centerY + offsetY));
            for (size_t i = 0; i < batchSize; i++)
            {
                Tensor<HWC, C3, U8> inputLeftCrop(
                    inputWidth, inputHeight,
                    const_cast<uint8_t *>(inputLeft.getData()) + i * inputLeft.getStride(TensorDimension::DEPTH),
                    false);
                Tensor<HWC, C3, U8> outputLeftCrop(
                    m_outputWidth, m_outputHeight,
                    m_resizedDeviceLeftInput.getData() + i * m_resizedDeviceLeftInput.getStride(TensorDimension::DEPTH),
                    false);
                Tensor<HWC, C3, U8> inputRightCrop(
                    inputWidth, inputHeight,
                    const_cast<uint8_t *>(inputRight.getData()) + i * inputRight.getStride(TensorDimension::DEPTH),
                    false);
                Tensor<HWC, C3, U8> outputRightCrop(m_outputWidth, m_outputHeight,
                                                    m_resizedDeviceRightInput.getData() +
                                                        i * m_resizedDeviceRightInput.getStride(TensorDimension::DEPTH),
                                                    false);
                tensor_ops::CropAndResize(outputLeftCrop, inputLeftCrop, dstCrop, srcCrop,
                                          tensor_ops::InterpolationType::INTERP_LINEAR, stream);
                tensor_ops::CropAndResize(outputRightCrop, inputRightCrop, dstCrop, srcCrop,
                                          tensor_ops::InterpolationType::INTERP_LINEAR, stream);
            }
        }

        if (m_swapRB)
        {
            tensor_ops::ConvertColorFormat(m_resizedDeviceLeftInput, m_resizedDeviceLeftInput, tensor_ops::BGR2RGB,
                                           stream);
            tensor_ops::ConvertColorFormat(m_resizedDeviceRightInput, m_resizedDeviceRightInput, tensor_ops::BGR2RGB,
                                           stream);
        }

        float scale[3];
        for (size_t i = 0; i < 3; i++)
        {
            scale[i] = m_preProcessorParams.normalization[i] / m_preProcessorParams.stdDev[i];
        }

        tensor_ops::Normalize(m_normalizedDeviceLeftInput, m_resizedDeviceLeftInput, scale,
                              m_preProcessorParams.pixelMean, stream);
        tensor_ops::Normalize(m_normalizedDeviceRightInput, m_resizedDeviceRightInput, scale,
                              m_preProcessorParams.pixelMean, stream);
        tensor_ops::InterleavedToPlanar(outputLeft, m_normalizedDeviceLeftInput, stream);
        tensor_ops::InterleavedToPlanar(outputRight, m_normalizedDeviceRightInput, stream);
    }
};

void ESSPreProcessor::execute(Tensor<NCHW, C3, F32> &outputLeft, Tensor<NCHW, C3, F32> &outputRight,
                              const Tensor<NHWC, C3, U8> &inputLeft, const Tensor<NHWC, C3, U8> &inputRight,
                              cudaStream_t stream)
{
#ifdef NVBENCH_ENABLE
    const std::string testName = "ESSPreProcessor_batch" + std::to_string(inputLeft.getDepth()) + "_" +
                                 std::to_string(inputLeft.getWidth()) + "x" + std::to_string(inputLeft.getHeight()) +
                                 "_GPU";
    nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
    m_pImpl->process(outputLeft, outputRight, inputLeft, inputRight, stream);
}

ESSPreProcessor::ESSPreProcessor(const ImagePreProcessingParams &preProcessorParams,
                                 const ModelInputParams &modelInputParams, const ESSPreProcessorParams &essParams)
    : m_pImpl(new ESSPreProcessor::ESSPreProcessorImpl(preProcessorParams, modelInputParams, essParams))
{
}

ESSPreProcessor::~ESSPreProcessor() {}

}} // namespace cvcore::ess
