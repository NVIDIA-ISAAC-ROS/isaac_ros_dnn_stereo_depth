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

#include "isaac_ros_dnn_stereo_decoder/dnn_stereo_decoder_node.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

namespace
{
constexpr uint8_t kDLPackFloat = 2;

void ValidateFloatTensor(const Tensor & tensor, const std::string & name)
{
  if (tensor.dtype_code != kDLPackFloat || tensor.dtype_bits != 32 ||
    tensor.dtype_lanes != 1)
  {
    throw std::invalid_argument(
            "[DNNStereoDecoderNode] Tensor '" + name + "' must be float32");
  }
  if (tensor.shape.size() != 3 && tensor.shape.size() != 4) {
    throw std::invalid_argument(
            "[DNNStereoDecoderNode] Tensor '" + name + "' must have rank 3 or 4");
  }
  for (const int64_t dim : tensor.shape) {
    if (dim <= 0) {
      throw std::invalid_argument(
              "[DNNStereoDecoderNode] Tensor '" + name + "' dimensions must be positive");
    }
  }
}

}  // namespace

DNNStereoDecoderNode::DNNStereoDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("dnn_stereo_decoder_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  cache_camera_info_{declare_parameter<bool>("cache_camera_info", false)},
  tensor_sub_{},
  camera_info_sub_{},
  sync_{ExactPolicy{3}, tensor_sub_, camera_info_sub_},
  disparity_tensor_name_{declare_parameter<std::string>(
      "disparity_tensor_name",
      "disparity")},
  confidence_tensor_name_{declare_parameter<std::string>(
      "confidence_tensor_name",
      "")},
  min_disparity_{declare_parameter<double>("min_disparity", 0.0)},
  max_disparity_{declare_parameter<double>("max_disparity", 10000.0)},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.0)}
{
  CHECK_CUDA_ERROR(
    nvidia::isaac_ros::common::initNamedCudaStream(stream_, "dnn_stereo_decoder_node"),
    "Failed to initialize CUDA stream");

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  disparity_pub_ = create_publisher<stereo_msgs::msg::DisparityImage>(
    "disparity", output_qos_, pub_options);

  // Validate confidence_threshold_ is within range [0, 1]
  if (confidence_threshold_ < 0.0 || confidence_threshold_ > 1.0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "confidence_threshold must be in range [0.0, 1.0], got: %f",
      confidence_threshold_);
    throw std::invalid_argument("confidence_threshold out of range [0.0, 1.0]");
  }

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";

  if (cache_camera_info_) {
    // Caching mode: Subscribe to topics independently without synchronization
    // Camera info is cached and every tensor message is processed immediately
    RCLCPP_INFO(
      this->get_logger(),
      "Camera info caching enabled - processing every tensor message with cached camera info");

    camera_info_sub_cached_mode_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "right/camera_info", input_qos_,
      std::bind(&DNNStereoDecoderNode::CameraInfoCallback, this, std::placeholders::_1),
      sub_options);

    tensor_sub_cached_mode_ = create_subscription<TensorList>(
      "tensor_sub", input_qos_,
      std::bind(&DNNStereoDecoderNode::TensorCallback, this, std::placeholders::_1),
      sub_options);
  } else {
    // Synchronization mode: Use exact time synchronization
    // Only process synchronized messages, discard unsynchronized with warning
    RCLCPP_INFO(
      this->get_logger(),
      "Using exact time synchronization - unsynchronized messages will be discarded");

    tensor_sub_.subscribe(this, "tensor_sub", input_qos_, sub_options);
    camera_info_sub_.subscribe(this, "right/camera_info", input_qos_, sub_options);

    sync_.registerCallback(
      std::bind(
        &DNNStereoDecoderNode::SynchronizedCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    sync_.getPolicy()->registerDropCallback(
      std::bind(
        &DNNStereoDecoderNode::UnsynchronizedCallback, this,
        std::placeholders::_1, std::placeholders::_2));
  }
}

void DNNStereoDecoderNode::SynchronizedCallback(
  const TensorList::ConstSharedPtr & tensor_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Processing synchronized tensor and camera info pair!");

  ProcessTensorAndCameraInfo(tensor_msg, camera_info_msg);
}

void DNNStereoDecoderNode::UnsynchronizedCallback(
  const TensorList::ConstSharedPtr & tensor_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // In synchronization mode, unsynchronized messages are discarded with a warning
  if (!tensor_msg && !camera_info_msg) {
    RCLCPP_WARN(
      this->get_logger(),
      "Both tensor and camera info messages missing - skipping the frame");
  } else if (!tensor_msg) {
    RCLCPP_WARN(
      this->get_logger(),
      "Tensor message missing - skipping the frame");
  } else if (!camera_info_msg) {
    RCLCPP_WARN(
      this->get_logger(),
      "Camera info message missing - skipping the frame");
  } else {
    RCLCPP_WARN(
      this->get_logger(),
      "Tensor and camera info pair dropped due to timestamp mismatch - "
      "consider enabling 'cache_camera_info' parameter to process all tensor messages");
  }
}

void DNNStereoDecoderNode::CameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  stored_camera_info_ = camera_info_msg;
  RCLCPP_DEBUG(this->get_logger(), "Camera info cached");
}

void DNNStereoDecoderNode::TensorCallback(
  const TensorList::ConstSharedPtr & tensor_msg)
{
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_info = stored_camera_info_;
  }

  if (!camera_info) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "Tensor received but no camera info cached yet - waiting for camera info");
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "Processing tensor with cached camera info");
  ProcessTensorAndCameraInfo(tensor_msg, camera_info);
}

void DNNStereoDecoderNode::ProcessTensorAndCameraInfo(
  const TensorList::ConstSharedPtr & tensor_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // Guard against degenerate projection matrix that would make baseline undefined
  if (std::abs(camera_info_msg->p[0]) <= std::numeric_limits<double>::epsilon()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Camera info focal_length_x (p[0]=%f) is zero/near-zero; skipping frame",
      camera_info_msg->p[0]);
    return;
  }

  const Tensor * disparity_tensor =
    isaac_ros_tensor_msgs::FindTensorByName(*tensor_msg, disparity_tensor_name_);
  if (!disparity_tensor) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Tensor '%s' not found in tensor list", disparity_tensor_name_.c_str());
    return;
  }

  try {
    ValidateFloatTensor(*disparity_tensor, disparity_tensor_name_);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(this->get_logger(), "%s", error.what());
    return;
  }

  if (!isaac_ros_tensor_msgs::IsContiguousRowMajor(*disparity_tensor)) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Disparity tensor '%s' must be contiguous row-major",
      disparity_tensor_name_.c_str());
    return;
  }

  const size_t rank = disparity_tensor->shape.size();
  const int dynamic_height_dim = ComputeHeightDim(rank);
  const int dynamic_width_dim = ComputeWidthDim(rank);
  const uint32_t height = static_cast<uint32_t>(disparity_tensor->shape[dynamic_height_dim]);
  const uint32_t width = static_cast<uint32_t>(disparity_tensor->shape[dynamic_width_dim]);
  const size_t image_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) *
    sizeof(float);

  size_t storage_elements = 0;
  try {
    storage_elements = isaac_ros_tensor_msgs::RequiredStorageElements(*disparity_tensor);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(this->get_logger(), "%s", error.what());
    return;
  }
  const size_t tensor_bytes = storage_elements * sizeof(float);
  if (tensor_bytes < image_bytes) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Disparity tensor buffer too small for %ux%u image", width, height);
    return;
  }
  if (disparity_tensor->byte_offset > disparity_tensor->data.size() ||
    tensor_bytes > disparity_tensor->data.size() -
    static_cast<size_t>(disparity_tensor->byte_offset))
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Disparity tensor '%s' buffer is too small", disparity_tensor_name_.c_str());
    return;
  }

  auto disparity_msg = std::make_unique<stereo_msgs::msg::DisparityImage>();
  disparity_msg->header = tensor_msg->header;
  disparity_msg->image.header = tensor_msg->header;
  disparity_msg->image.height = height;
  disparity_msg->image.width = width;
  disparity_msg->image.encoding = "32FC1";
  disparity_msg->image.is_bigendian = false;
  disparity_msg->image.step = width * static_cast<uint32_t>(sizeof(float));
  disparity_msg->f = static_cast<float>(camera_info_msg->p[0]);
  disparity_msg->t = static_cast<float>(-camera_info_msg->p[3] / camera_info_msg->p[0]);
  disparity_msg->min_disparity = static_cast<float>(min_disparity_);
  disparity_msg->max_disparity = static_cast<float>(max_disparity_);
  disparity_msg->delta_d = 0.0f;
  disparity_msg->valid_window.x_offset = 0;
  disparity_msg->valid_window.y_offset = 0;
  disparity_msg->valid_window.width = width;
  disparity_msg->valid_window.height = height;

  disparity_msg->image.data = cuda_buffer_backend::allocate_buffer(image_bytes);
  auto output_handle =
    cuda_buffer_backend::from_output_buffer(disparity_msg->image.data, stream_);
  float * gpu_data = reinterpret_cast<float *>(output_handle.get_ptr());

  try {
    auto input_handle =
      cuda_buffer_backend::from_input_buffer(disparity_tensor->data, stream_);
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        gpu_data,
        input_handle.get_ptr() + disparity_tensor->byte_offset,
        image_bytes, cudaMemcpyDeviceToDevice, stream_),
      "Failed to copy disparity tensor to output buffer");
  } catch (const std::exception & error) {
    RCLCPP_ERROR(this->get_logger(), "%s", error.what());
    return;
  }

  // Apply confidence masking if confidence map exists and threshold > 0.
  if (confidence_threshold_ > 0.0) {
    const std::string conf_name = confidence_tensor_name_;
    if (!conf_name.empty()) {
      const Tensor * conf_tensor =
        isaac_ros_tensor_msgs::FindTensorByName(*tensor_msg, conf_name);
      if (!conf_tensor) {
        RCLCPP_WARN(
          this->get_logger(),
          "Confidence tensor '%s' not found; skipping confidence masking",
          conf_name.c_str());
      } else {
        try {
          ValidateFloatTensor(*conf_tensor, conf_name);
          if (!isaac_ros_tensor_msgs::IsContiguousRowMajor(*conf_tensor)) {
            throw std::invalid_argument("Confidence tensor must be contiguous row-major");
          }
          const size_t conf_rank = conf_tensor->shape.size();
          const uint32_t conf_height =
            static_cast<uint32_t>(conf_tensor->shape[ComputeHeightDim(conf_rank)]);
          const uint32_t conf_width =
            static_cast<uint32_t>(conf_tensor->shape[ComputeWidthDim(conf_rank)]);
          if (conf_height != height || conf_width != width) {
            throw std::invalid_argument("Confidence tensor dims mismatch");
          }
          const size_t conf_bytes =
            isaac_ros_tensor_msgs::RequiredStorageElements(*conf_tensor) * sizeof(float);
          if (conf_tensor->byte_offset > conf_tensor->data.size() ||
            conf_bytes > conf_tensor->data.size() -
            static_cast<size_t>(conf_tensor->byte_offset))
          {
            throw std::invalid_argument("Confidence tensor buffer is too small");
          }

          auto conf_handle =
            cuda_buffer_backend::from_input_buffer(conf_tensor->data, stream_);
          const cudaError_t conf_err =
            nvidia::isaac_ros::dnn_stereo_decoder::ApplyConfidenceThreshold(
            gpu_data,
            reinterpret_cast<const float *>(
              conf_handle.get_ptr() + conf_tensor->byte_offset),
            width, height,
            static_cast<float>(confidence_threshold_), stream_);
          CHECK_CUDA_ERROR(conf_err, "CUDA error after ApplyConfidenceThreshold kernel");
        } catch (const std::exception & error) {
          RCLCPP_WARN(
            this->get_logger(),
            "Skipping confidence masking: %s", error.what());
        }
      }
    }
  }

  const cudaError_t filter_err =
    nvidia::isaac_ros::dnn_stereo_decoder::FilterDisparity(
    gpu_data, width, height,
    static_cast<float>(min_disparity_), static_cast<float>(max_disparity_),
    stream_);
  CHECK_CUDA_ERROR(filter_err, "CUDA error after FilterDisparity kernel");

  CHECK_CUDA_ERROR(
    cudaStreamSynchronize(stream_),
    "Failed to synchronize CUDA stream before publishing");

  disparity_pub_->publish(std::move(disparity_msg));
}

DNNStereoDecoderNode::~DNNStereoDecoderNode()
{
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_), "Failed to destroy CUDA stream");
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode)
