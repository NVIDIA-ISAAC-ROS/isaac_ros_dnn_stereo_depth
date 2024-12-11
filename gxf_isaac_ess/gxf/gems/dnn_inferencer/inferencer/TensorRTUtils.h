/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */
#pragma once

#include "NvInferRuntime.h"
#include "extensions/tensorops/core/Tensor.h"
#include "Errors.h"

namespace cvcore {
namespace inferencer {

/*
 * Maps tensorrt datatype to cvcore Channel type.
 * @param channelType cvcore channel type.
 * @param dtype tensorrt datatype
 * return error code
 */
std::error_code getCVCoreChannelTypeFromTensorRT(
    cvcore::tensor_ops::ChannelType& channelType,
    nvinfer1::DataType dtype);

/*
 * Maps tensorrt datatype to cvcore Channel type.
 * @param channelLayout cvcore channel type.
 * @param dtype tensorrt layout
 * return error code
 */
std::error_code getCVCoreChannelLayoutFromTensorRT(
    cvcore::tensor_ops::TensorLayout& channelLayout,
    nvinfer1::TensorFormat tensorFormat);

}  // namespace inferencer
}  // namespace cvcore
