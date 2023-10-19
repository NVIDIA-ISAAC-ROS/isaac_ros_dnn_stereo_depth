/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "video_buffer_thresholder.cu.hpp"

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "engine/gems/cuda_utils/launch_utils.hpp"

namespace nvidia {
namespace isaac {
namespace {

__constant__ uint32_t NUM_CHANNELS = 3;

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
                          const float fill_value) {
  cudaMemset(output, 0, image_size.x * image_size.y * sizeof(float));
  dim3 block(16, 16);
  dim3 grid(DivRoundUp(image_size.x, 16), DivRoundUp(image_size.y, 16), 1);
  threshold_image_cuda_kernel << < grid, block >> > (image, mask, output, threshold, fill_value,
                                                    image_size);
}

void threshold_image_cuda(const unsigned char * image, const float * mask,
                          unsigned char * output, const int2 image_size, const float threshold,
                          const uchar3 fill_value) {
  cudaMemset(output, 0, image_size.x * image_size.y * sizeof(uint8_t) * NUM_CHANNELS);
  dim3 block(16, 16);
  dim3 grid(DivRoundUp(image_size.x, 16), DivRoundUp(image_size.y, 16), 1);
  threshold_image_cuda_kernel << < grid, block >> > (image, mask, output, threshold, fill_value,
                                                    image_size);
}

}  // namespace isaac
}  // namespace nvidia
