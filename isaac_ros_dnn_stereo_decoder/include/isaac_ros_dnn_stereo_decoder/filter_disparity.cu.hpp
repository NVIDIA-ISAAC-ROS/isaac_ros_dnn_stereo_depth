// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DNN_STEREO_DECODER__FILTER_DISPARITY_CU_HPP_
#define ISAAC_ROS_DNN_STEREO_DECODER__FILTER_DISPARITY_CU_HPP_

#include <cstdint>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_decoder
{

// In-place filter that sets disparity to 0 when not finite or
// outside [min_disparity, max_disparity]
cudaError_t FilterDisparity(
  float * disparity, const uint32_t width, const uint32_t height,
  const float min_disparity, const float max_disparity, const cudaStream_t stream);

// In-place filter that zeroes disparity where confidence < threshold
cudaError_t ApplyConfidenceThreshold(
  float * disparity, const float * confidence,
  const uint32_t width, const uint32_t height,
  const float threshold, const cudaStream_t stream);

}  // namespace dnn_stereo_decoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_STEREO_DECODER__FILTER_DISPARITY_CU_HPP_
