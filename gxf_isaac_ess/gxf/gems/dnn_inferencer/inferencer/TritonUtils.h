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

#ifdef ENABLE_TRITON
#include <string>

#include "grpc_client.h"

#include "cv/core/Tensor.h"
#include "cv/inferencer/Errors.h"

namespace cvcore {
namespace inferencer {

/*
 * Maps triton datatype to cvcore Channel type.
 * @param channelType cvcore channel type.
 * @param dtype String representing triton datatype
 * return bool returns false if mapping was not successful.
 */
bool getCVCoreChannelType(cvcore::ChannelType& channelType, std::string dtype);

/*
 * Maps triton datatype to cvcore Channel type.
 * @param dtype String representing triton datatype
 * @param channelType cvcore channel type.
 * return bool returns false if mapping was not successful.
 */
bool getTritonChannelType(std::string& dtype, cvcore::ChannelType channelType);

}  // namespace inferencer
}  // namespace cvcore
#endif  // ENABLE_TRITON
