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

#pragma once

#ifdef ENABLE_TRITON
#include <grpc_client.h>
#include <string>
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
