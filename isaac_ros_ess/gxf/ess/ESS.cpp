// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/ess/ESS.hpp"

#include <algorithm>
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace cvcore {

namespace detail {

// Function to bind a cuda stream with cid into downstream message
gxf_result_t BindCudaStream(gxf::Entity& message, gxf_uid_t cid) {
  if (cid == kNullUid) {
    GXF_LOG_ERROR("stream_cid is null");
    return GXF_FAILURE;
  }
  auto output_stream_id = message.add<gxf::CudaStreamId>("stream");
  if (!output_stream_id) {
    GXF_LOG_ERROR("failed to add cudastreamid.");
    return GXF_FAILURE;
  }
  output_stream_id.value()->stream_cid = cid;
  return GXF_SUCCESS;
}

// Function to record a new cuda event
gxf_result_t RecordCudaEvent(gxf::Entity& message, gxf::Handle<gxf::CudaStream>& stream) {
  // Create a new event
  cudaEvent_t cuda_event;
  cudaEventCreateWithFlags(&cuda_event, 0);
  gxf::CudaEvent event;
  auto ret = event.initWithEvent(cuda_event, stream->dev_id(), [](auto) {});
  if (!ret) {
    GXF_LOG_ERROR("failed to init cuda event");
    return GXF_FAILURE;
  }
  // Record the event
  // Can define []() { GXF_LOG_DEBUG("tensorops event synced"); } as callback func for debug purpose
  ret = stream->record(event.event().value(),
                       [event = cuda_event, entity = message.clone().value()](auto) { cudaEventDestroy(event); });
  if (!ret) {
    GXF_LOG_ERROR("record event failed");
    return ret.error();
  }
  return GXF_SUCCESS;
}

} // namespace detail

gxf_result_t ESS::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(left_image_name_, "left_image_name", "The name of the left image to be received", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(right_image_name_, "right_image_name", "The name of the right image to be received",
                                 "", gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_name_, "output_name", "The name of the tensor to be passed to next node", "");
  result &= registrar->parameter(stream_pool_, "stream_pool", "The Cuda Stream pool for allocate cuda stream", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(pool_, "pool", "Memory pool for allocating output data", "");
  result &= registrar->parameter(left_image_receiver_, "left_image_receiver", "Receiver to get the left image", "");
  result &= registrar->parameter(right_image_receiver_, "right_image_receiver", "Receiver to get the right image", "");
  result &= registrar->parameter(output_transmitter_, "output_transmitter", "Transmitter to send the data", "");
  result &= registrar->parameter(output_adapter_, "output_adapter", "Adapter to send output data", "");

  result &= registrar->parameter(image_type_, "image_type", "Type of input image: BGR_U8 or RGB_U8", "");
  result &= registrar->parameter(pixel_mean_, "pixel_mean", "The mean for each channel", "");
  result &= registrar->parameter(normalization_, "normalization", "The normalization for each channel", "");
  result &=
    registrar->parameter(standard_deviation_, "standard_deviation", "The standard deviation for each channel", "");

  result &= registrar->parameter(max_batch_size_, "max_batch_size", "The max batch size to run inference on", "");
  result &= registrar->parameter(input_layer_width_, "input_layer_width", "The model input layer width", "");
  result &= registrar->parameter(input_layer_height_, "input_layer_height", "The model input layer height", "");
  result &= registrar->parameter(model_input_type_, "model_input_type", "The model input image: BGR_U8 or RGB_U8", "");

  result &= registrar->parameter(engine_file_path_, "engine_file_path", "The path to the serialized TRT engine", "");
  result &= registrar->parameter(input_layers_name_, "input_layers_name", "The names of the input layers", "");
  result &= registrar->parameter(output_layers_name_, "output_layers_name", "The names of the output layers", "");

  result &= registrar->parameter(preprocess_type_, "preprocess_type",
                                 "The type of ESS preprocessing: RESIZE / CENTER_CROP", "");
  result &= registrar->parameter(output_width_, "output_width", "The width of output result", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_height_, "output_height", "The height of output result", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(timestamp_policy_, "timestamp_policy",
                                 "Input channel to get timestamp 0(left)/1(right)", "", 0);

  return gxf::ToResultCode(result);
}

gxf_result_t ESS::start() {
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
  if (pixel_mean_vec.size() != 3 || normalization_vec.size() != 3 || standard_deviation_vec.size() != 3) {
    GXF_LOG_ERROR("Invalid preprocessing params.");
    return GXF_FAILURE;
  }

  if (image_type_.get() == "BGR_U8") {
    preProcessorParams.imgType = ::cvcore::BGR_U8;
  } else if (image_type_.get() == "RGB_U8") {
    preProcessorParams.imgType = ::cvcore::RGB_U8;
  } else {
    GXF_LOG_INFO("Wrong input image type. BGR_U8 and RGB_U8 are only supported.");
    return GXF_FAILURE;
  }
  std::copy(pixel_mean_vec.begin(), pixel_mean_vec.end(), preProcessorParams.pixelMean);
  std::copy(normalization_vec.begin(), normalization_vec.end(), preProcessorParams.normalization);
  std::copy(standard_deviation_vec.begin(), standard_deviation_vec.end(), preProcessorParams.stdDev);

  // Setting model input params for ESS
  modelInputParams.maxBatchSize     = max_batch_size_.get();
  modelInputParams.inputLayerWidth  = input_layer_width_.get();
  modelInputParams.inputLayerHeight = input_layer_height_.get();
  if (model_input_type_.get() == "BGR_U8") {
    modelInputParams.modelInputType = ::cvcore::BGR_U8;
  } else if (model_input_type_.get() == "RGB_U8") {
    modelInputParams.modelInputType = ::cvcore::RGB_U8;
  } else {
    GXF_LOG_INFO("Wrong model input type. BGR_U8 and RGB_U8 are only supported.");
    return GXF_FAILURE;
  }

  // Setting inference params for ESS
  inferenceParams = {engine_file_path_.get(), input_layers_name_.get(), output_layers_name_.get()};

  // Setting extra params for ESS
  if (preprocess_type_.get() == "RESIZE") {
    extraParams = {::cvcore::ess::PreProcessType::RESIZE};
  } else if (preprocess_type_.get() == "CENTER_CROP") {
    extraParams = {::cvcore::ess::PreProcessType::CENTER_CROP};
  } else {
    GXF_LOG_ERROR("Invalid preprocessing type.");
    return GXF_FAILURE;
  }

  // Setting ESS object with the provided params
  objESS.reset(new ::cvcore::ess::ESS(preProcessorParams, modelInputParams, inferenceParams, extraParams));

  return GXF_SUCCESS;
}

gxf_result_t ESS::tick() {
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
  if (cuda_stream != 0) {
    detail::RecordCudaEvent(inputLeftMessage.value(), cuda_stream_);
    auto inputLeftStreamId = inputLeftMessage.value().get<gxf::CudaStreamId>("stream");
    if (inputLeftStreamId) {
      auto inputLeftStream = gxf::Handle<gxf::CudaStream>::Create(inputLeftStreamId.value().context(),
                                                                  inputLeftStreamId.value()->stream_cid);
      // NOTE: This is an expensive call. It will halt the current CPU thread until all events
      //   previously associated with the stream are cleared
      if (!inputLeftStream.value()->syncStream()) {
        GXF_LOG_ERROR("sync left stream failed.");
        return GXF_FAILURE;
      }
    }
  }

  auto inputRightMessage = right_image_receiver_->receive();
  if (!inputRightMessage) {
    return GXF_FAILURE;
  }
  if (cuda_stream != 0) {
    detail::RecordCudaEvent(inputRightMessage.value(), cuda_stream_);
    auto inputRightStreamId = inputRightMessage.value().get<gxf::CudaStreamId>("stream");
    if (inputRightStreamId) {
      auto inputRightStream = gxf::Handle<gxf::CudaStream>::Create(inputRightStreamId.value().context(),
                                                                   inputRightStreamId.value()->stream_cid);
      // NOTE: This is an expensive call. It will halt the current CPU thread until all events
      //   previously associated with the stream are cleared
      if (!inputRightStream.value()->syncStream()) {
        GXF_LOG_ERROR("sync right stream failed.");
        return GXF_FAILURE;
      }
    }
  }

  auto maybeLeftName   = left_image_name_.try_get();
  auto leftInputBuffer = inputLeftMessage.value().get<gxf::VideoBuffer>(maybeLeftName ? maybeLeftName.value().c_str() : nullptr);
  if (!leftInputBuffer) {
    return GXF_FAILURE;
  }
  auto maybeRightName   = right_image_name_.try_get();
  auto rightInputBuffer = inputRightMessage.value().get<gxf::VideoBuffer>(maybeRightName ? maybeRightName.value().c_str() : nullptr);
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
  if (leftBufferInfo.width != rightBufferInfo.width || leftBufferInfo.height != rightBufferInfo.height ||
      leftBufferInfo.color_format != rightBufferInfo.color_format) {
    GXF_LOG_ERROR("left/right images mismatch.");
    return GXF_FAILURE;
  }
  const size_t outputWidth  = output_width_.try_get() ? output_width_.try_get().value() : leftBufferInfo.width;
  const size_t outputHeight = output_height_.try_get() ? output_height_.try_get().value() : leftBufferInfo.height;

  // Creating GXF Tensor or VideoBuffer to hold the data to be transmitted
  gxf::Expected<gxf::Entity> outputMessage = gxf::Entity::New(context());
  if (!outputMessage) {
    return outputMessage.error();
  }
  auto error = output_adapter_.get()->AddImageToMessage<::cvcore::ImageType::Y_F32>(
    outputMessage.value(), outputWidth, outputHeight, pool_.get(), false, output_name_.get().c_str());
  if (error != GXF_SUCCESS) {
    return GXF_FAILURE;
  }
  auto outputImage = output_adapter_.get()->WrapImageFromMessage<::cvcore::ImageType::Y_F32>(
    outputMessage.value(), output_name_.get().c_str());
  if (!outputImage) {
    return GXF_FAILURE;
  }

  // Creating CVCore Tensors to hold the input and output data
  ::cvcore::Tensor<::cvcore::NHWC, ::cvcore::C1, ::cvcore::F32> outputImageDevice(outputWidth, outputHeight, 1,
                                                                                  outputImage.value().getData(), false);

  // Running the inference
  auto inputColorFormat = leftBufferInfo.color_format;
  if (inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB ||
      inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR) {
    ::cvcore::Tensor<::cvcore::NHWC, ::cvcore::C3, ::cvcore::U8> leftImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<uint8_t*>(leftInputBuffer.value()->pointer()),
      false);
    ::cvcore::Tensor<::cvcore::NHWC, ::cvcore::C3, ::cvcore::U8> rightImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<uint8_t*>(rightInputBuffer.value()->pointer()),
      false);
    objESS->execute(outputImageDevice, leftImageDevice, rightImageDevice, cuda_stream);
  } else if (inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32 ||
             inputColorFormat == gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32) {
    ::cvcore::Tensor<::cvcore::NCHW, ::cvcore::C3, ::cvcore::F32> leftImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<float*>(leftInputBuffer.value()->pointer()),
      false);
    ::cvcore::Tensor<::cvcore::NCHW, ::cvcore::C3, ::cvcore::F32> rightImageDevice(
      leftBufferInfo.width, leftBufferInfo.height, 1, reinterpret_cast<float*>(rightInputBuffer.value()->pointer()),
      false);
    objESS->execute(outputImageDevice, leftImageDevice, rightImageDevice, cuda_stream);
  } else {
    GXF_LOG_ERROR("invalid input image type.");
    return GXF_FAILURE;
  }

  // Allocate a cuda event that can be used to record on each tick
  if (!cuda_stream_.is_null()) {
    detail::BindCudaStream(outputMessage.value(), cuda_stream_.cid());
    detail::RecordCudaEvent(outputMessage.value(), cuda_stream_);
  }

  // Pass down timestamp if necessary
  auto maybeDaqTimestamp = timestamp_policy_.get() == 0 ? inputLeftMessage.value().get<gxf::Timestamp>()
                                                        : inputRightMessage.value().get<gxf::Timestamp>();
  if (maybeDaqTimestamp) {
    auto outputTimestamp = outputMessage.value().add<gxf::Timestamp>(maybeDaqTimestamp.value().name());
    if (!outputTimestamp) {
      return outputTimestamp.error();
    }
    *outputTimestamp.value() = *maybeDaqTimestamp.value();
  }

  // Send the data
  output_transmitter_->publish(outputMessage.value());
  return GXF_SUCCESS;
}

gxf_result_t ESS::stop() {
  objESS.reset(nullptr);
  return GXF_SUCCESS;
}

} // namespace cvcore
} // namespace nvidia
