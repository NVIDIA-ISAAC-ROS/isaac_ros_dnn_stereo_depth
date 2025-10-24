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
#include <algorithm>
#include <stdexcept>

#include "extensions/ess/inference/ESS.h"
#include "extensions/tensorops/core/BBoxUtils.h"
#include "extensions/tensorops/core/CVError.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"

namespace nvidia {
namespace isaac {
namespace ess {
using ImageType = cvcore::tensor_ops::ImageType;
using BBox = cvcore::tensor_ops::BBox;

struct ESSPreProcessor::ESSPreProcessorImpl {
    size_t m_maxBatchSize;
    size_t m_outputWidth;
    size_t m_outputHeight;
    PreProcessType m_processType;
    ImagePreProcessingParams m_preProcessorParams;
    Tensor_NHWC_C3_U8 m_resizedDeviceLeftInput;
    Tensor_NHWC_C3_U8 m_resizedDeviceRightInput;
    Tensor_NHWC_C3_F32 m_normalizedDeviceLeftInput;
    Tensor_NHWC_C3_F32 m_normalizedDeviceRightInput;
    bool m_swapRB;

    ESSPreProcessorImpl(const ImagePreProcessingParams & imgParams,
        const ModelInputParams & modelParams,
        size_t output_width, size_t output_height,
        const ESSPreProcessorParams & essParams)
        : m_maxBatchSize(modelParams.maxBatchSize)
        , m_outputHeight(output_height)
        , m_outputWidth(output_width)
        , m_processType(essParams.preProcessType)
        , m_preProcessorParams(imgParams) {
        if (imgParams.imgType != ImageType::BGR_U8 &&
            imgParams.imgType != ImageType::RGB_U8) {
            throw std::invalid_argument("ESSPreProcessor : Only image types RGB_U8/BGR_U8 are"
                "supported\n");
        }
        m_resizedDeviceLeftInput = {output_width, output_height,
            modelParams.maxBatchSize, false};
        m_resizedDeviceRightInput    = {output_width, output_height,
                                     modelParams.maxBatchSize, false};
        m_normalizedDeviceLeftInput  = {output_width, output_height,
                                       modelParams.maxBatchSize, false};
        m_normalizedDeviceRightInput = {output_width, output_height,
                                        modelParams.maxBatchSize, false};
        m_swapRB                     = imgParams.imgType != modelParams.modelInputType;
    }

    void process(Tensor_NCHW_C3_F32 & outputLeft, Tensor_NCHW_C3_F32 & outputRight,
        const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
        cudaStream_t stream) {
        if (inputLeft.isCPU() || inputRight.isCPU() || outputLeft.isCPU() ||
            outputRight.isCPU()) {
            throw std::invalid_argument("ESSPreProcessor : Input/Output Tensor must be"
                "GPU Tensor.");
        }

        if (outputLeft.getWidth() != m_outputWidth || outputLeft.getHeight() != m_outputHeight ||
            outputRight.getWidth() != m_outputWidth || outputRight.getHeight() != m_outputHeight) {
            throw std::invalid_argument(
                "ESSPreProcessor : Output Tensor dimension does not match network input"
                "requirement");
        }

        if (inputLeft.getWidth() != inputRight.getWidth() ||
            inputLeft.getHeight() != inputRight.getHeight()) {
            throw std::invalid_argument("ESSPreProcessor : Input tensor dimensions don't match");
        }

        if (outputLeft.getDepth() != inputLeft.getDepth() ||
            outputRight.getDepth() != inputRight.getDepth() ||
            inputLeft.getDepth() != inputRight.getDepth()) {
            throw std::invalid_argument("ESSPreProcessor : Input/Output Tensor batchsize"
                "mismatch.");
        }

        if (outputLeft.getDepth() > m_maxBatchSize) {
            throw std::invalid_argument("ESSPreProcessor : Input/Output batchsize exceeds"
                "max batch size.");
        }

        const size_t batchSize   = inputLeft.getDepth();
        const size_t inputWidth  = inputLeft.getWidth();
        const size_t inputHeight = inputLeft.getHeight();

        if (m_processType == PreProcessType::RESIZE) {
            cvcore::tensor_ops::Resize(m_resizedDeviceLeftInput, inputLeft,
                false, cvcore::tensor_ops::INTERP_LINEAR, stream);
            cvcore::tensor_ops::Resize(m_resizedDeviceRightInput, inputRight,
                false, cvcore::tensor_ops::INTERP_LINEAR, stream);
        } else {
            const float centerX = static_cast<float>(inputWidth) / 2.0;
            const float centerY = static_cast<float>(inputHeight) / 2.0;
            const float offsetX = static_cast<float>(m_outputWidth) / 2.0;
            const float offsetY = static_cast<float>(m_outputHeight) / 2.0;
            BBox srcCrop, dstCrop;
            dstCrop = {0, 0, static_cast<int>(m_outputWidth - 1), static_cast<int>(
                m_outputHeight - 1)};
            srcCrop.xmin = std::max(0, static_cast<int>(centerX - offsetX));
            srcCrop.ymin = std::max(0, static_cast<int>(centerY - offsetY));
            srcCrop.xmax = std::min(static_cast<int>(m_outputWidth - 1),
                static_cast<int>(centerX + offsetX));
            srcCrop.ymax = std::min(static_cast<int>(m_outputHeight - 1),
                static_cast<int>(centerY + offsetY));
            for (size_t i = 0; i < batchSize; i++) {
                Tensor_HWC_C3_U8 inputLeftCrop(
                    inputWidth, inputHeight,
                    const_cast<uint8_t *>(inputLeft.getData()) + i *
                        inputLeft.getStride(TensorDimension::DEPTH),
                    false);
                Tensor_HWC_C3_U8 outputLeftCrop(
                    m_outputWidth, m_outputHeight,
                    m_resizedDeviceLeftInput.getData() +
                        i * m_resizedDeviceLeftInput.getStride(TensorDimension::DEPTH),
                    false);
                Tensor_HWC_C3_U8 inputRightCrop(
                    inputWidth, inputHeight,
                    const_cast<uint8_t *>(inputRight.getData()) +
                        i * inputRight.getStride(TensorDimension::DEPTH),
                    false);
                Tensor_HWC_C3_U8 outputRightCrop(m_outputWidth, m_outputHeight,
                    m_resizedDeviceRightInput.getData() +
                        i * m_resizedDeviceRightInput.getStride(TensorDimension::DEPTH),
                    false);
                cvcore::tensor_ops::CropAndResize(outputLeftCrop, inputLeftCrop, dstCrop, srcCrop,
                    cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
                cvcore::tensor_ops::CropAndResize(outputRightCrop, inputRightCrop, dstCrop,
                    srcCrop, cvcore::tensor_ops::InterpolationType::INTERP_LINEAR, stream);
            }
        }

        if (m_swapRB) {
            cvcore::tensor_ops::ConvertColorFormat(m_resizedDeviceLeftInput,
                m_resizedDeviceLeftInput, cvcore::tensor_ops::BGR2RGB, stream);
            cvcore::tensor_ops::ConvertColorFormat(m_resizedDeviceRightInput,
                m_resizedDeviceRightInput, cvcore::tensor_ops::BGR2RGB, stream);
        }

        float scale[3];
        for (size_t i = 0; i < 3; i++) {
            scale[i] = m_preProcessorParams.normalization[i] / m_preProcessorParams.stdDev[i];
        }

        cvcore::tensor_ops::Normalize(m_normalizedDeviceLeftInput, m_resizedDeviceLeftInput,
            scale, m_preProcessorParams.pixelMean, stream);
        cvcore::tensor_ops::Normalize(m_normalizedDeviceRightInput, m_resizedDeviceRightInput,
            scale, m_preProcessorParams.pixelMean, stream);
        cvcore::tensor_ops::InterleavedToPlanar(outputLeft, m_normalizedDeviceLeftInput, stream);
        cvcore::tensor_ops::InterleavedToPlanar(outputRight, m_normalizedDeviceRightInput, stream);
    }
};

void ESSPreProcessor::execute(Tensor_NCHW_C3_F32 & outputLeft,
    Tensor_NCHW_C3_F32 & outputRight,
    const Tensor_NHWC_C3_U8 & inputLeft, const Tensor_NHWC_C3_U8 & inputRight,
    cudaStream_t stream) {
    m_pImpl->process(outputLeft, outputRight, inputLeft, inputRight, stream);
}

ESSPreProcessor::ESSPreProcessor(const ImagePreProcessingParams & preProcessorParams,
    const ModelInputParams & modelInputParams,
    const size_t output_width, const size_t output_height,
    const ESSPreProcessorParams & essParams)
    : m_pImpl(new ESSPreProcessor::ESSPreProcessorImpl(preProcessorParams,
        modelInputParams, output_width, output_height, essParams)) {}

ESSPreProcessor::~ESSPreProcessor() {}

}  // namespace ess
}  // namespace isaac
}  // namespace nvidia
