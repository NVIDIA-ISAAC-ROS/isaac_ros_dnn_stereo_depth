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

#include "extensions/video_buffer_utils/components/video_buffer_thresholder.hpp"

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "extensions/messages/camera_message.hpp"
#include "extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.hpp"
#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gems/video_buffer/allocator.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"

namespace nvidia {
namespace isaac {

gxf_result_t VideoBufferThresholder::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      image_input_, "image_input", "Input image",
      "Incoming VideoBuffer image to be masked.");
  result &= registrar->parameter(
      mask_input_, "mask_input", "Input Mask",
      "Incoming VideoBuffer mask. It must have the same dimension (with, height) as input image.");
  result &= registrar->parameter(
      threshold_, "threshold", "Threshold",
      "Value used to threshold image pixels based on the mask value, i.e., "
      "if mask[index] <= threshold: image[index] = fill_value.");
  result &= registrar->parameter(
      fill_value_float_, "fill_value_float", "Fill Value float",
      "Value to fill the masked pixels, to be used only for float input images.", 0.0f);
  result &= registrar->parameter(
      fill_value_rgb_, "fill_value_rgb", "Fill Value RGB",
      "Value to fill the masked pixels, to be used only for RGB input images.", Vector3i(0, 0, 0));
  result &= registrar->parameter(
      masked_output_, "masked_output", "Masked output image",
      "Masked output image, after threshold.");
  result &= registrar->parameter(
      allocator_, "allocator", "Allocator",
      "Allocator to allocate output messages.");
  result &= registrar->parameter(
      video_buffer_name_, "video_buffer_name", "Video Buffer Name",
      "Name of the VideoBuffer component to use.",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      stream_pool_, "stream_pool", "Stream pool",
      "The Cuda Stream pool for allocate cuda stream",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

gxf_result_t VideoBufferThresholder::start() {
  // Allocate cuda stream using stream pool if necessary
  if (stream_pool_.try_get()) {
    auto stream = stream_pool_.try_get().value()->allocateStream();
    if (!stream) {
      GXF_LOG_ERROR("allocating stream failed.");
      return GXF_FAILURE;
    }
    cuda_stream_ = std::move(stream.value());
    if (!cuda_stream_->stream()) {
      GXF_LOG_ERROR("allocated stream is not initialized.");
      return GXF_FAILURE;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoBufferThresholder::stop() {
  return GXF_SUCCESS;
}

gxf::Expected<void> AddInputTimestampToOutput(gxf::Entity& output, gxf::Entity input) {
  std::string timestamp_name{"timestamp"};
  auto maybe_timestamp = input.get<gxf::Timestamp>(timestamp_name.c_str());

  // Default to unnamed
  if (!maybe_timestamp) {
    maybe_timestamp = input.get<gxf::Timestamp>();
  }

  if (!maybe_timestamp) {
    GXF_LOG_ERROR("Failed to get input timestamp!");
    return gxf::ForwardError(maybe_timestamp);
  }

  auto maybe_out_timestamp = output.add<gxf::Timestamp>(timestamp_name.c_str());
  if (!maybe_out_timestamp) {
    GXF_LOG_ERROR("Failed to add timestamp to output message!");
    return gxf::ForwardError(maybe_out_timestamp);
  }

  *maybe_out_timestamp.value() = *maybe_timestamp.value();
  return gxf::Success;
}

gxf_result_t VideoBufferThresholder::tick() {
  // Read input message and validate them
  gxf::Entity image_entity = UNWRAP_OR_RETURN(image_input_.get()->receive());
  gxf::Entity mask_entity = UNWRAP_OR_RETURN(mask_input_.get()->receive());
  gxf::Entity output_entity;

  // Get a CUDA stream for execution
  cudaStream_t cuda_stream = 0;
  if (!cuda_stream_.is_null()) {
    cuda_stream = cuda_stream_->stream().value();
  }

  gxf::Handle<gxf::VideoBuffer> mask_video_buffer =
      UNWRAP_OR_RETURN(mask_entity.get<gxf::VideoBuffer>());
  gxf::Handle<gxf::VideoBuffer> image_video_buffer, tx_video_buffer;

  if (!IsCameraMessage(image_entity).value()) {
    auto video_buffer_name = video_buffer_name_.try_get();
    const char* name = video_buffer_name ? video_buffer_name->c_str() : nullptr;
    image_video_buffer = UNWRAP_OR_RETURN(image_entity.get<gxf::VideoBuffer>(name));
    const gxf::VideoBufferInfo& video_info = image_video_buffer->video_frame_info();
    output_entity = UNWRAP_OR_RETURN(gxf::Entity::New(context()));
    tx_video_buffer = UNWRAP_OR_RETURN(output_entity.add<gxf::VideoBuffer>(name));
    RETURN_IF_ERROR(validateImageMessage(image_video_buffer));
    if (video_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
      RETURN_IF_ERROR(AllocateUnpaddedVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F>(
        tx_video_buffer, video_info.width, video_info.height,
        gxf::MemoryStorageType::kDevice, allocator_));
    } else if (video_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB) {
        RETURN_IF_ERROR(AllocateUnpaddedVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>(
        tx_video_buffer, video_info.width, video_info.height,
        gxf::MemoryStorageType::kDevice, allocator_));
    }
    RETURN_IF_ERROR(AddInputTimestampToOutput(output_entity, image_entity));
  } else {
    CameraMessageParts image_message = UNWRAP_OR_RETURN(GetCameraMessage(image_entity));
    image_video_buffer = image_message.frame;
    RETURN_IF_ERROR(validateImageMessage(image_video_buffer));
    gxf::Expected<CameraMessageParts> camera_message = CreateCameraMessage(
        context(), image_video_buffer->video_frame_info(), image_video_buffer->size(),
        static_cast<gxf::MemoryStorageType>(image_video_buffer->storage_type()), allocator_);
    if (!camera_message) {
      return gxf::ToResultCode(camera_message);
    }

    *camera_message->intrinsics = *image_message.intrinsics;
    *camera_message->extrinsics = *image_message.extrinsics;
    *camera_message->sequence_number = *image_message.sequence_number;
    *camera_message->timestamp = *image_message.timestamp;
    tx_video_buffer = camera_message->frame;
    output_entity = camera_message->entity;
  }

  // Validate video buffer and perform thresholding
  RETURN_IF_ERROR(validateMaskMessage(image_video_buffer, mask_video_buffer));
  RETURN_IF_ERROR(thresholdImage(image_video_buffer, mask_video_buffer,
    tx_video_buffer, cuda_stream));
  cudaError_t result = cudaStreamSynchronize(cuda_stream);
  if (result != cudaSuccess) {
    GXF_LOG_ERROR("Error while synchronizing stream: %s", cudaGetErrorString(result));
    return GXF_FAILURE;
  }
  RETURN_IF_ERROR(gxf::ToResultCode(masked_output_->publish(output_entity)));

  return GXF_SUCCESS;
}

gxf_result_t VideoBufferThresholder::thresholdImage(
  const gxf::Handle<gxf::VideoBuffer> image_video_buffer,
  const gxf::Handle<gxf::VideoBuffer> mask_video_buffer,
  gxf::Handle<gxf::VideoBuffer> output_video_buffer,
  cudaStream_t cuda_stream) const {
  const gxf::VideoBufferInfo& video_info = image_video_buffer->video_frame_info();
  uint32_t buffer_width = video_info.width;
  uint32_t buffer_height = video_info.height;

  // Perform thresholding
  if (video_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB) {
    threshold_image_cuda(
      reinterpret_cast<const unsigned char*>(image_video_buffer->pointer()),
      reinterpret_cast<const float*>(mask_video_buffer->pointer()),
      reinterpret_cast<unsigned char*>(output_video_buffer->pointer()),
      {static_cast<int>(buffer_width), static_cast<int>(buffer_height)},
      threshold_.get(),
      toUchar3(fill_value_rgb_.get()),
      cuda_stream);
  } else if (video_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
    threshold_image_cuda(
      reinterpret_cast<const float*>(image_video_buffer->pointer()),
      reinterpret_cast<const float*>(mask_video_buffer->pointer()),
      reinterpret_cast<float*>(output_video_buffer->pointer()),
      {static_cast<int>(buffer_width), static_cast<int>(buffer_height)},
      threshold_.get(),
      fill_value_float_.get(),
      cuda_stream);
  } else {
    return GXF_INVALID_DATA_FORMAT;
  }

  return GXF_SUCCESS;
}

uchar3 VideoBufferThresholder::toUchar3(const Vector3i& vector3i) const {
  return {
      static_cast<unsigned char>(vector3i.x()),
      static_cast<unsigned char>(vector3i.y()),
      static_cast<unsigned char>(vector3i.z())};
}

gxf_result_t VideoBufferThresholder::validateImageMessage(
    const gxf::Handle<gxf::VideoBuffer> image) const {
  if (image->storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Input image must be stored in "
                  "gxf::MemoryStorageType::kDevice");
    return GXF_INVALID_DATA_FORMAT;
  }

  const gxf::VideoBufferInfo image_info = image->video_frame_info();
  if (image_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB &&
      image_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
    GXF_LOG_ERROR("Input image must be of type "
                  "gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F or"
                  "gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA");
    return GXF_INVALID_DATA_FORMAT;
  }

  return GXF_SUCCESS;
}

gxf_result_t VideoBufferThresholder::validateMaskMessage(
    const gxf::Handle<gxf::VideoBuffer> image,
    const gxf::Handle<gxf::VideoBuffer> mask) const {
  if (image->storage_type() != mask->storage_type()) {
    GXF_LOG_ERROR("Input image image must be stored in the same type as input depth");
    return GXF_INVALID_DATA_FORMAT;
  }

  const gxf::VideoBufferInfo mask_info = mask->video_frame_info();
  if (mask_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F &&
      mask_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
    GXF_LOG_ERROR("Input mask image must be of type "
                  "gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F or"
                  "gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F");
    return GXF_INVALID_DATA_FORMAT;
  }

  const gxf::VideoBufferInfo image_info = image->video_frame_info();
  if (mask_info.width != image_info.width || mask_info.height != image_info.height) {
    GXF_LOG_ERROR("Input image dimensions must match input mask dimensions");
    return GXF_INVALID_DATA_FORMAT;
  }
  return GXF_SUCCESS;
}

}  // namespace isaac
}  // namespace nvidia
