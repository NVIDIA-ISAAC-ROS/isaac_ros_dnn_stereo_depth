// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.hpp"

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"

#include "gems/cuda_utils/launch_utils.hpp"

namespace nvidia {
namespace isaac {
namespace {

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

CONSTANT uint32_t NUM_CHANNELS = 3;

__device__ float get_pixel(const float * input)
{
  return *input;
}

__device__ void set_pixel(const float pixel, float* output)
{
  *output = pixel;
}

__global__ void threshold_image_cuda_kernel(
    const float * image, const float * mask, float * output,
    float threshold, float fill_value, int2 image_size)
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < image_size.x && y < image_size.y)
  {
    const uint32_t index_mask = (y * image_size.x + x);
    const uint32_t index_image = index_mask;
    float pixel_out =  (mask[index_mask] > threshold)? get_pixel(image + index_image) : fill_value;
    set_pixel(pixel_out, output + index_image);
  }
}

__device__ uchar3 get_pixel_uchar3(const unsigned char * input)
{
  return make_uchar3(input[0], input[1], input[2]);
}

__device__ void set_pixel_uchar3(const uchar3 pixel, unsigned char * output)
{
  output[0] = pixel.x;
  output[1] = pixel.y;
  output[2] = pixel.z;
}

__global__ void threshold_image_cuda_kernel(
    const unsigned char * image, const float * mask, unsigned char * output,
    float threshold, uchar3 fill_value, int2 image_size)
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < image_size.x && y < image_size.y)
  {
    const uint32_t index_mask = (y * image_size.x + x);
    const uint32_t index_image = index_mask * NUM_CHANNELS;
    uchar3 pixel_out = (mask[index_mask] > threshold) ?
      get_pixel_uchar3(image + index_image) : fill_value;
    set_pixel_uchar3(pixel_out, output + index_image);
  }
}

}  // namespace

void threshold_image_cuda(const float* image, const float* mask,
                          float* output, const int2 image_size, const float threshold,
                          const float fill_value, cudaStream_t cuda_stream) {
  cudaMemsetAsync(output, 0, image_size.x * image_size.y * sizeof(float), cuda_stream);
  dim3 block(16, 16);
  dim3 grid(DivRoundUp(image_size.x, 16), DivRoundUp(image_size.y, 16), 1);
  threshold_image_cuda_kernel << < grid, block, 0, cuda_stream >> > (image, mask, output, threshold, fill_value,
                                                    image_size);
}

void threshold_image_cuda(const unsigned char * image, const float * mask,
                          unsigned char * output, const int2 image_size, const float threshold,
                          const uchar3 fill_value, cudaStream_t cuda_stream) {
  cudaMemsetAsync(output, 0, image_size.x * image_size.y * sizeof(uint8_t) * NUM_CHANNELS, cuda_stream);
  dim3 block(16, 16);
  dim3 grid(DivRoundUp(image_size.x, 16), DivRoundUp(image_size.y, 16), 1);
  threshold_image_cuda_kernel << < grid, block, 0, cuda_stream >> > (image, mask, output, threshold, fill_value,
                                                    image_size);
}

}  // namespace isaac
}  // namespace nvidia
