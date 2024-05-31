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

#include "gems/dnn_inferencer/inferencer/TensorRTUtils.h"
#include <iostream>

namespace cvcore {
namespace inferencer {
using TensorLayout = cvcore::tensor_ops::TensorLayout;
using ChannelType = cvcore::tensor_ops::ChannelType;
using ChannelCount =  cvcore::tensor_ops::ChannelCount;
using ErrorCode = cvcore::tensor_ops::ErrorCode;

std::error_code getCVCoreChannelTypeFromTensorRT(ChannelType& channelType,
    nvinfer1::DataType dtype) {
    if (dtype == nvinfer1::DataType::kINT8) {
        channelType = ChannelType::U8;
    } else if (dtype == nvinfer1::DataType::kHALF) {
        channelType = ChannelType::F16;
    } else if (dtype == nvinfer1::DataType::kFLOAT) {
        channelType = ChannelType::F32;
    } else {
        return ErrorCode::INVALID_OPERATION;
    }

    return ErrorCode::SUCCESS;
}

std::error_code getCVCoreChannelLayoutFromTensorRT(TensorLayout& channelLayout,
    nvinfer1::TensorFormat tensorFormat) {
    if (tensorFormat == nvinfer1::TensorFormat::kLINEAR) {
        channelLayout = TensorLayout::NCHW;
    } else if (tensorFormat == nvinfer1::TensorFormat::kHWC) {
        channelLayout = TensorLayout::HWC;
    } else {
        return ErrorCode::INVALID_OPERATION;
    }

    return ErrorCode::SUCCESS;
}

}  // namespace inferencer
}  // namespace cvcore
