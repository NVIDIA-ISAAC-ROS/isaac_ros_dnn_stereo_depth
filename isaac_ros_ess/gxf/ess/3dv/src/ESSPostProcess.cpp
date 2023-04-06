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

struct ESSPostProcessor::ESSPostProcessorImpl
{
    ESSPostProcessorImpl(const ModelInputParams &modelParams)
        : m_maxBatchSize(modelParams.maxBatchSize)
        , m_networkWidth(modelParams.inputLayerWidth)
        , m_networkHeight(modelParams.inputLayerHeight)
    {
        m_scaledDisparityDevice = {m_networkWidth, m_networkHeight, m_maxBatchSize, false};
        m_outputDisparityDevice = {m_networkWidth, m_networkHeight, m_maxBatchSize, false};
    }

    void resizeBuffers(std::size_t width, std::size_t height)
    {
        if (m_outputDisparityDevice.getWidth() == width && m_outputDisparityDevice.getHeight() == height)
        {
            return;
        }
        m_outputDisparityDevice = {width, height, m_maxBatchSize, false};
    }

    void process(Tensor<NHWC, C1, F32> &outputDisparity, const Tensor<NHWC, C1, F32> &inputDisparity,
                 cudaStream_t stream)
    {
        if (inputDisparity.isCPU())
        {
            throw std::invalid_argument("ESSPostProcessor : Input Tensor must be GPU Tensor.");
        }

        if (inputDisparity.getWidth() != m_networkWidth || inputDisparity.getHeight() != m_networkHeight)
        {
            throw std::invalid_argument(
                "ESSPostProcessor : Input Tensor dimension "
                "does not match network input "
                "requirement");
        }

        if (inputDisparity.getDepth() != outputDisparity.getDepth())
        {
            throw std::invalid_argument("ESSPostProcessor : Input/Output Tensor batchsize mismatch.");
        }

        const size_t batchSize = inputDisparity.getDepth();
        if (batchSize > m_maxBatchSize)
        {
            throw std::invalid_argument("ESSPostProcessor : Input batchsize exceeds Max Batch size.");
        }
        const size_t outputWidth  = outputDisparity.getWidth();
        const size_t outputHeight = outputDisparity.getHeight();

        // Disparity map values are scaled based on the outputWidth/networkInputWidth ratio
        const float scale = static_cast<float>(outputWidth) / m_networkWidth;
        Tensor<NHWC, C1, F32> scaledDisparity(m_scaledDisparityDevice.getWidth(), m_scaledDisparityDevice.getHeight(),
                                              batchSize, m_scaledDisparityDevice.getData(), false);

        tensor_ops::Normalize(scaledDisparity, inputDisparity, scale, 0, stream);
        if (!outputDisparity.isCPU())
        {
            tensor_ops::Resize(outputDisparity, m_scaledDisparityDevice, stream);
        }
        else
        {
            resizeBuffers(outputWidth, outputHeight);
            Tensor<NHWC, C1, F32> outputDisparityDevice(m_outputDisparityDevice.getWidth(),
                                                        m_outputDisparityDevice.getHeight(), batchSize,
                                                        m_outputDisparityDevice.getData(), false);
            tensor_ops::Resize(outputDisparityDevice, m_scaledDisparityDevice, stream);
            cvcore::Copy(outputDisparity, outputDisparityDevice, stream);
            CHECK_ERROR(cudaStreamSynchronize(stream));
        }
    }

    size_t m_maxBatchSize;
    size_t m_networkWidth, m_networkHeight;
    Tensor<NHWC, C1, F32> m_scaledDisparityDevice;
    Tensor<NHWC, C1, F32> m_outputDisparityDevice;
};

void ESSPostProcessor::execute(Tensor<NHWC, C1, F32> &outputDisparity, const Tensor<NHWC, C1, F32> &inputDisparity,
                               cudaStream_t stream)
{
#ifdef NVBENCH_ENABLE
    const std::string testName = "ESSPostProcessor_batch" + std::to_string(outputDisparity.getDepth()) + "_" +
                                 std::to_string(outputDisparity.getWidth()) + "x" +
                                 std::to_string(outputDisparity.getHeight()) + "_GPU";
    nv::bench::Timer timerFunc = nv::bench::GPU(testName.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
    m_pImpl->process(outputDisparity, inputDisparity, stream);
}

ESSPostProcessor::ESSPostProcessor(const ModelInputParams &modelInputParams)
    : m_pImpl(new ESSPostProcessor::ESSPostProcessorImpl(modelInputParams))
{
}

ESSPostProcessor::~ESSPostProcessor() {}

}} // namespace cvcore::ess
