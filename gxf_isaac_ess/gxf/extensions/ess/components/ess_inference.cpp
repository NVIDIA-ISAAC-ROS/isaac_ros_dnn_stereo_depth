// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/ess/components/ess_inference.hpp"

#include <dlfcn.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <utility>

#include "extensions/tensorops/components/ImageUtils.hpp"
#include "extensions/tensorops/core/Core.h"
#include "extensions/tensorops/core/Image.h"
#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gems/hash/hash_file.hpp"
#include "gems/video_buffer/allocator.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac {

namespace detail {

template<typename T>
gxf_result_t PassthroughComponents(gxf::Entity& output, gxf::Entity& input,
    const char* name = nullptr) {
  auto maybe_component = input.get<T>(name);
  if (maybe_component) {
    auto output_component = output.add<T>(name != nullptr ? name : maybe_component.value().name());
    if (!output_component) {
      GXF_LOG_ERROR("add output component failed.");
      return output_component.error();
    }
    *(output_component.value()) = *(maybe_component.value());
  } else {
    GXF_LOG_DEBUG("component %s not found.", name);
  }

  return GXF_SUCCESS;
}

gxf::Expected<std::string> ComputeEnginePath(
    const ess::ModelBuildParams& modelBuildParams) {
  const SHA256::String onnx_hash =
    UNWRAP_OR_RETURN(hash_file(modelBuildParams.onnx_file_path.c_str()));

  std::string target_dir = "/tmp";

  char* test_tmpdir = std::getenv("TEST_TMPDIR");

  if (test_tmpdir) {
    target_dir = test_tmpdir;
  } else {
    char* tmpdir = std::getenv("TMPDIR");

    if (tmpdir) {
      target_dir = tmpdir;
    }
  }

  std::stringstream path;

  path << target_dir << "/engine_" << onnx_hash.c_str() <<
    "_" << modelBuildParams.enable_fp16 <<
    "_" << modelBuildParams.max_workspace_size <<
    "_" << modelBuildParams.dla_core <<
    "_" << ess::InferencerVersion <<
    ".engine";

  return path.str();
}

}  // namespace detail

gxf_result_t ESSInference::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      left_image_name_, "left_image_name", "Left image name",
      "The name of the left image to be received",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      right_image_name_, "right_image_name", "Right image name",
      "The name of the right image to be received",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      output_name_, "output_name", "Output name",
      "The name of the tensor to be passed to next node");
  result &= registrar->parameter(
      stream_pool_, "stream_pool", "Stream pool",
      "The Cuda Stream pool for allocate cuda stream",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      pool_, "pool", "Pool",
      "Memory pool for allocating output data");
  result &= registrar->parameter(
      left_image_receiver_, "left_image_receiver", "Left image receiver",
      "Receiver to get the left image");
  result &= registrar->parameter(
      right_image_receiver_, "right_image_receiver", "Right image receiver",
      "Receiver to get the right image");
  result &= registrar->parameter(
      output_transmitter_, "output_transmitter", "Output transmitter",
      "Transmitter to send the data");
  result &= registrar->parameter(
      confidence_transmitter_, "confidence_transmitter", "Confidence transmitter",
      "Transmitter to send the confidence data");
  result &= registrar->parameter(
      image_type_, "image_type", "Image type",
      "Type of input image: BGR_U8 or RGB_U8");
  result &= registrar->parameter(
      pixel_mean_, "pixel_mean", "Pixel mean",
      "The mean for each channel");
  result &= registrar->parameter(
      normalization_, "normalization", "Normalization",
      "The normalization for each channel");
  result &= registrar->parameter(
      standard_deviation_, "standard_deviation", "Standard deviation",
      "The standard deviation for each channel");
  result &= registrar->parameter(
      max_batch_size_, "max_batch_size", "Max batch size",
      "The max batch size to run inference on");
  result &= registrar->parameter(
      model_input_type_, "model_input_type", "Model input type",
      "The model input image: BGR_U8 or RGB_U8");

  result &= registrar->parameter(
      force_engine_update_, "force_engine_update", "Force engine update",
      "Flag that indicates always update engine", false);
  result &= registrar->parameter(
      onnx_file_path_, "onnx_file_path", "ONNX file path",
      "The path to the onnx model file");
  result &= registrar->parameter(
      tensorrt_plugin_path_, "tensorrt_plugin", "TensorRT plugin path",
      "The path to the TensorRT plugin file",
      gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      enable_fp16_, "enable_fp16", "Enable FP16",
      "Flag to enable FP16 in engine generation", true);
  result &= registrar->parameter(
      max_workspace_size_, "max_workspace_size", "Max Workspace Size",
      "Size of working space in bytes. Default to 64MB", 64L * (1L << 20));
  result &= registrar->parameter(
      dla_core_, "dla_core", "DLA Core",
      "DLA Core to use. Fallback to GPU is always enabled. "
      "Default to use GPU only.",
      gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      engine_file_path_, "engine_file_path", "Engine file path",
      "The path to the serialized TRT engine (default is to auto-generated unique name)",
      gxf::Registrar::NoDefaultParameter(),
      GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      input_layers_name_, "input_layers_name", "Input layer names",
      "The names of the input layers");
  result &= registrar->parameter(
      output_layers_name_, "output_layers_name", "Output layer names",
      "The names of the output layers");

  result &= registrar->parameter(
      preprocess_type_, "preprocess_type", "Preprocess type",
      "The type of ESS preprocessing: RESIZE / CENTER_CROP");
  result &= registrar->parameter(
      timestamp_policy_, "timestamp_policy", "Timestamp policy",
      "Input channel to get timestamp 0(left)/1(right)", 0);

  return gxf::ToResultCode(result);
}

gxf_result_t ESSInference::start() {
  using ImageType = cvcore::tensor_ops::ImageType;

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
  // Setting image pre-processing params for ESS
  const auto& pixel_mean_vec         = pixel_mean_.get();
  const auto& normalization_vec      = normalization_.get();
  const auto& standard_deviation_vec = standard_deviation_.get();
  if (pixel_mean_vec.size() != 3 || normalization_vec.size() != 3 ||
      standard_deviation_vec.size() != 3) {
    GXF_LOG_ERROR("Invalid preprocessing params.");
    return GXF_FAILURE;
  }
  if (image_type_.get() == "BGR_U8") {
    preprocessor_params_.imgType = ImageType::BGR_U8;
  } else if (image_type_.get() == "RGB_U8") {
    preprocessor_params_.imgType = ImageType::RGB_U8;
  } else {
    GXF_LOG_INFO("Wrong input image type. BGR_U8 and RGB_U8 are only supported.");
    return GXF_FAILURE;
  }
  std::copy(pixel_mean_vec.begin(), pixel_mean_vec.end(), preprocessor_params_.pixelMean);
  std::copy(normalization_vec.begin(),
      normalization_vec.end(),
      preprocessor_params_.normalization);
  std::copy(standard_deviation_vec.begin(),
      standard_deviation_vec.end(),
      preprocessor_params_.stdDev);

  // Setting model input params for ESS
  model_input_params_.maxBatchSize = max_batch_size_.get();
  if (model_input_type_.get() == "BGR_U8") {
    model_input_params_.modelInputType = ImageType::BGR_U8;
  } else if (model_input_type_.get() == "RGB_U8") {
    model_input_params_.modelInputType = ImageType::RGB_U8;
  } else {
    GXF_LOG_INFO("Wrong model input type. BGR_U8 and RGB_U8 are only supported.");
    return GXF_FAILURE;
  }

  // Load ESS plugin
  if (tensorrt_plugin_path_.try_get() && !tensorrt_plugin_path_.try_get().value().empty()) {
    if(!dlopen(tensorrt_plugin_path_.try_get().value().c_str(), RTLD_NOW)) {
      GXF_LOG_ERROR("ESS plugin loading failed.");
      return GXF_FAILURE;
    }
  }

  // Setting engine build params
  auto maybe_dla_core = dla_core_.try_get();
  const int64_t dla_core = dla_core_.try_get().value_or(-1);

  model_build_params_ = {force_engine_update_.get(), onnx_file_path_.get(),
    enable_fp16_.get(), max_workspace_size_.get(), dla_core};

  gxf::Expected<std::string> maybe_engine_file_path = engine_file_path_.try_get();
  std::string engine_file_path;

  if (maybe_engine_file_path) {
    engine_file_path = maybe_engine_file_path.value();
  } else {
    engine_file_path =
      UNWRAP_OR_RETURN(detail::ComputeEnginePath(model_build_params_));
  }

  // Setting inference params for ESS
  inference_params_ = {
     engine_file_path, input_layers_name_.get(), output_layers_name_.get()};

  // Setting extra params for ESS
  if (preprocess_type_.get() == "RESIZE") {
    extra_params_ = {ess::PreProcessType::RESIZE};
  } else if (preprocess_type_.get() == "CENTER_CROP") {
    extra_params_ = {ess::PreProcessType::CENTER_CROP};
  } else {
    GXF_LOG_ERROR("Invalid preprocessing type.");
    return GXF_FAILURE;
  }

  // Setting ESS object with the provided params
  ess_.reset(new ess::ESS(preprocessor_params_, model_input_params_,
    model_build_params_, inference_params_, extra_params_));

  return GXF_SUCCESS;
}

gxf_result_t ESSInference::tick() {
  using ImageType = cvcore::tensor_ops::ImageType;
  using TensorLayout = cvcore::tensor_ops::TensorLayout;
  using ChannelType = cvcore::tensor_ops::ChannelType;
  using ChannelCount =  cvcore::tensor_ops::ChannelCount;
  using Tensor_NHWC_C1_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC,
    ChannelCount::C1, ChannelType::F32>;
  using Tensor_NCHW_C3_F32 = cvcore::tensor_ops::Tensor<TensorLayout::NCHW,
    ChannelCount::C3, ChannelType::F32>;
  using Tensor_NHWC_C3_U8 = cvcore::tensor_ops::Tensor<TensorLayout::NHWC,
    ChannelCount::C3, ChannelType::U8>;

  // Get a CUDA stream for execution
  cudaStream_t cuda_stream = 0;
  if (!cuda_stream_.is_null()) {
    cuda_stream = cuda_stream_->stream().value();
  }
  // Receiving the data
  auto inputLeftMessage = left_image_receiver_->receive();
  if (!inputLeftMessage) {
    return GXF_FAILURE;
  }

  auto inputRightMessage = right_image_receiver_->receive();
  if (!inputRightMessage) {
    return GXF_FAILURE;
  }

  auto maybeLeftName   = left_image_name_.try_get();
  auto leftInputBuffer = inputLeftMessage.value().get<gxf::VideoBuffer>(
      maybeLeftName ? maybeLeftName.value().c_str() : nullptr);
  if (!leftInputBuffer) {
    return GXF_FAILURE;
  }
  auto maybeRightName   = right_image_name_.try_get();
  auto rightInputBuffer = inputRightMessage.value().get<gxf::VideoBuffer>(
      maybeRightName ? maybeRightName.value().c_str() : nullptr);
  if (!rightInputBuffer) {
    return GXF_FAILURE;
  }
  if (leftInputBuffer.value()->storage_type() != gxf::MemoryStorageType::kDevice ||
      rightInputBuffer.value()->storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("input images must be on GPU.");
    return GXF_FAILURE;
  }

  const auto& leftBufferInfo  = leftInputBuffer.value()->video_frame_info();
  const auto& rightBufferInfo = rightInputBuffer.value()->video_frame_info();
  if (leftBufferInfo.width != rightBufferInfo.width ||
      leftBufferInfo.height != rightBufferInfo.height ||
      leftBufferInfo.color_format != rightBufferInfo.color_format) {
    GXF_LOG_ERROR("left/right images mismatch.");
    return GXF_FAILURE;
  }
  const size_t outputWidth  = ess_->getModelOutputWidth();
  const size_t outputHeight = ess_->getModelOutputHeight();

  // Creating GXF Tensor or VideoBuffer to hold the data to be transmitted
  gxf::Expected<gxf::Entity> outputMessage = gxf::Entity::New(context());
  if (!outputMessage) {
    return outputMessage.error();
  }

  // Creating GXF tensor to hold the data to be transmitted
  auto videoBuffer = outputMessage->add<gxf::VideoBuffer>(output_name_.get().c_str());
  if (!videoBuffer) {
    GXF_LOG_ERROR("ESS::tick ==> Failed to create video buffer in output message.");
    return videoBuffer.error();
  }
  auto result = AllocateUnpaddedVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F>(
    videoBuffer.value(), outputWidth, outputHeight, gxf::MemoryStorageType::kDevice, pool_);
  if (!result) {
    GXF_LOG_ERROR("ESS::tick ==> Failed to allocate video buffer in output message.");
    return GXF_FAILURE;
  }
  using D = typename cvcore::tensor_ops::detail::ChannelTypeToNative<
      cvcore::tensor_ops::ImageTraits<ImageType::Y_F32, 3>::CT>::Type;
  const auto info = videoBuffer.value()->video_frame_info();
  auto pointer = reinterpret_cast<D*>(videoBuffer.value()->pointer());
  if (!pointer) {
    GXF_LOG_ERROR("ESS::tick ==> Failed to reinterpret video buffer pointer.");
    return GXF_FAILURE;
  }
  const auto& color_planes = info.color_planes;
  auto outputImage = cvcore::tensor_ops::Image<ImageType::Y_F32>(info.width, info.height,
      color_planes[0].stride, pointer, false);

  // Creating GXF VideoBuffer to hold confidence map to be transmitted
  gxf::Expected<gxf::Entity> outputConfMessage = gxf::Entity::New(context());
  if (!outputConfMessage) {
    return outputConfMessage.error();
  }
  auto confBuffer = outputConfMessage->add<gxf::VideoBuffer>(output_name_.get().c_str());
  if (!confBuffer) {
    return confBuffer.error();
  }
  result = AllocateUnpaddedVideoBuffer<gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32F>(
    confBuffer.value(), outputWidth, outputHeight, gxf::MemoryStorageType::kDevice, pool_);
  if (!result) {
    GXF_LOG_ERROR("ESS::tick ==> Failed to allocate video buffer in confidence message.");
    return GXF_FAILURE;
  }

  pointer = reinterpret_cast<D*>(confBuffer.value()->pointer());
  if (!pointer) {
    return GXF_FAILURE;
  }
  auto outputConf = cvcore::tensor_ops::Image<ImageType::Y_F32>(info.width, info.height,
      color_planes[0].stride, pointer, false);

  // Creating CVCore Tensors to hold the input and output data
  Tensor_NHWC_C1_F32 outputImageDevice(
    outputWidth, outputHeight, 1,
    outputImage.getData(), false);

  Tensor_NHWC_C1_F32 outputConfDevice(
    outputWidth, outputHeight, 1,
    outputConf.getData(), false);

  // Running the inference
  auto inputColorFormat = leftBufferInfo.color_format;
  if (inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB ||
      inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR) {
    Tensor_NHWC_C3_U8 leftImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<uint8_t*>(
        leftInputBuffer.value()->pointer()), false);
    Tensor_NHWC_C3_U8 rightImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<uint8_t*>
        (rightInputBuffer.value()->pointer()), false);
    ess_->execute(outputImageDevice, outputConfDevice, leftImageDevice,
      rightImageDevice, cuda_stream);
  } else if (inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32 ||
             inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32) {
    Tensor_NCHW_C3_F32 leftImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<float*>(
        leftInputBuffer.value()->pointer()), false);
    Tensor_NCHW_C3_F32 rightImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<float*>(
        rightInputBuffer.value()->pointer()), false);
    ess_->execute(outputImageDevice, outputConfDevice, leftImageDevice,
      rightImageDevice, cuda_stream);
  } else {
    GXF_LOG_ERROR("invalid input image type.");
    return GXF_FAILURE;
  }

  // Pass down timestamp if necessary
  auto maybeDaqTimestamp =
    timestamp_policy_.get() == 0 ? inputLeftMessage.value().get<gxf::Timestamp>()
                                 : inputRightMessage.value().get<gxf::Timestamp>();
  if (maybeDaqTimestamp) {
    auto outputTimestamp = outputMessage.value().add<gxf::Timestamp>(
      maybeDaqTimestamp.value().name());
    if (!outputTimestamp) {
      return outputTimestamp.error();
    }
    *outputTimestamp.value() = *maybeDaqTimestamp.value();

    auto confTimestamp = outputConfMessage.value().add<gxf::Timestamp>(
      maybeDaqTimestamp.value().name());
    if (!confTimestamp) {
      return confTimestamp.error();
    }
    *confTimestamp.value() = *maybeDaqTimestamp.value();
  }

  detail::PassthroughComponents<int64_t>(outputMessage.value(), inputLeftMessage.value(),
      "sequence_number");
  detail::PassthroughComponents<gxf::Pose3D>(outputMessage.value(), inputRightMessage.value(),
      "extrinsics");
  detail::PassthroughComponents<int64_t>(outputConfMessage.value(), inputLeftMessage.value(),
      "sequence_number");
  detail::PassthroughComponents<gxf::Pose3D>(outputConfMessage.value(), inputRightMessage.value(),
      "extrinsics");

  // use intrinsics scaling since input and output resolution may differ
  auto maybe_input_model = inputLeftMessage->get<gxf::CameraModel>("intrinsics");
  if (maybe_input_model) {
    auto maybe_output_model = outputMessage->add<gxf::CameraModel>("intrinsics");
    if (!maybe_output_model) {
      GXF_LOG_ERROR("creating output intrinsics failed.");
      return gxf::ToResultCode(maybe_output_model);
    }
    auto maybe_output_conf = outputConfMessage->add<gxf::CameraModel>("intrinsics");
    if (!maybe_output_conf) {
      GXF_LOG_ERROR("creating conf intrinsics failed.");
      return gxf::ToResultCode(maybe_output_conf);
    }
    gxf::Handle<gxf::CameraModel> input_model = maybe_input_model.value();
    gxf::Handle<gxf::CameraModel> output_model = maybe_output_model.value();
    gxf::Handle<gxf::CameraModel> output_conf = maybe_output_conf.value();

    gxf::Expected<gxf::CameraModel> maybe_scaled_model =
      tensor_ops::GetScaledCameraModel(*input_model, outputWidth, outputHeight, false);

    if (!maybe_scaled_model) {
      GXF_LOG_ERROR("computing output intrinsics failed.");
      return gxf::ToResultCode(maybe_scaled_model);
    }

    *output_model = *maybe_scaled_model;
    *output_conf = *maybe_scaled_model;
  } else {
    GXF_LOG_DEBUG("Input message is missing intrinsics!");
  }

  // Publish the data and confidence
  RETURN_IF_ERROR(gxf::ToResultCode(output_transmitter_->publish(outputMessage.value())));
  RETURN_IF_ERROR(gxf::ToResultCode(confidence_transmitter_->publish(
    outputConfMessage.value())));

  return GXF_SUCCESS;
}

gxf_result_t ESSInference::stop() {
  ess_.reset(nullptr);
  return GXF_SUCCESS;
}

}  // namespace isaac
}  // namespace nvidia
