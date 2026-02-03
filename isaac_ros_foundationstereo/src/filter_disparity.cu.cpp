// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_foundationstereo/filter_disparity.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationstereo
{
namespace
{

__global__ void FilterDisparityKernel(
  float * disparity, const uint32_t width, const uint32_t height,
  const float min_disp, const float max_disp)
{
  const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= width || v >= height) {
    return;
  }

  const uint32_t idx = v * width + u;
  float d = disparity[idx];

  // Filter: set to 0 if NaN, inf, or outside range
  if (!isfinite(d) || d < min_disp || d > max_disp) {
    disparity[idx] = 0.0f;
  }
}

}  // namespace

void FilterDisparity(
  float * disparity, const uint32_t width, const uint32_t height,
  const float min_disparity, const float max_disparity, const cudaStream_t stream)
{
  dim3 threads_per_block{32, 32, 1};
  dim3 blocks{(width + threads_per_block.x - 1) / threads_per_block.x,
              (height + threads_per_block.y - 1) / threads_per_block.y,
              1};
  FilterDisparityKernel<<<blocks, threads_per_block, 0, stream>>>(
    disparity, width, height, min_disparity, max_disparity);
}

}  // namespace foundationstereo
}  // namespace isaac_ros
}  // namespace nvidia
