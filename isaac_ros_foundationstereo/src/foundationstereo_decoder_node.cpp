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

#include "isaac_ros_foundationstereo/foundationstereo_decoder_node.hpp"
#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{
namespace
{

// Helper function to get timestamps from NitrosCameraInfo
std::pair<int32_t, uint32_t> GetCameraInfoTimestamps(
  const nvidia::isaac_ros::nitros::NitrosCameraInfo & nitros_camera_info)
{
  auto context = nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext();
  auto msg_entity = nvidia::gxf::Entity::Shared(context, nitros_camera_info.handle);

  auto input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>("timestamp");
  if (!input_timestamp) {
    input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>();
  }

  if (input_timestamp) {
    int32_t sec = static_cast<int32_t>(
      input_timestamp.value()->acqtime / static_cast<uint64_t>(1e9));
    uint32_t nanosec = static_cast<uint32_t>(
      input_timestamp.value()->acqtime % static_cast<uint64_t>(1e9));
    return {sec, nanosec};
  }

  return {0, 0};  // Default timestamps if not available
}

}  // namespace

FoundationStereoDecoderNode::FoundationStereoDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("foundationstereo_decoder_node", options),
  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  tensor_nitros_sub_{},
  camera_info_sub_{},
  sync_{ExactPolicy{3}, tensor_nitros_sub_, camera_info_sub_},
  nitros_pub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosDisparityImage>>(
    this, "disparity",
    nvidia::isaac_ros::nitros::nitros_disparity_image_32FC1_t::supported_type_name,
    nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, output_qos_)},
  disparity_tensor_name_{declare_parameter<std::string>(
      "disparity_tensor_name",
      "disparity")},
  min_disparity_{declare_parameter<double>("min_disparity", 0.0)},
  max_disparity_{declare_parameter<double>("max_disparity", 10000.0)}
{
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_), "Failed to create CUDA stream");

  // Subscribe to topics
  tensor_nitros_sub_.subscribe(this, "tensor_sub");
  camera_info_sub_.subscribe(this, "right/camera_info");

  // Register synchronized callback
  sync_.registerCallback(
    std::bind(
      &FoundationStereoDecoderNode::SynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Register drop callback for unsynchronized messages
  sync_.getPolicy()->registerDropCallback(
    std::bind(
      &FoundationStereoDecoderNode::UnsynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));
}

void FoundationStereoDecoderNode::SynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Processing synchronized tensor and camera info pair!");

  auto tensor_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*tensor_msg);
  auto camera_timestamps = GetCameraInfoTimestamps(*camera_info_msg);

  if (tensor_view.GetTimestampSeconds() != camera_timestamps.first ||
    tensor_view.GetTimestampNanoseconds() != camera_timestamps.second)
  {
    RCLCPP_WARN(this->get_logger(), "Both messages received, but timestamps didn't match!");
    return;
  }

  ProcessTensorAndCameraInfo(tensor_msg, camera_info_msg);
}

void FoundationStereoDecoderNode::UnsynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg)
{
  // Check for missing messages in unsynchronized pair
  if (!tensor_msg) {
    RCLCPP_WARN(this->get_logger(),
    "Tensor message is missing in unsynchronized pair - cannot process disparity");
    return;
  }

  if (!camera_info_msg) {
    RCLCPP_WARN(this->get_logger(),
    "Camera info message is missing in unsynchronized pair - cannot process disparity");
    return;
  }

  RCLCPP_WARN(this->get_logger(),
  "Received unsynchronized tensor and camera info pair - using unsynchronized camera parameters");

  ProcessTensorAndCameraInfo(tensor_msg, camera_info_msg);
}

void FoundationStereoDecoderNode::ProcessTensorAndCameraInfo(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg)
{
  auto tensor_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*tensor_msg);

  // Get tensor dimensions
  auto tensor = tensor_view.GetNamedTensor(disparity_tensor_name_);
  int height = tensor.GetDimension(height_dim_);
  int width = tensor.GetDimension(width_dim_);

  // Create header
  std_msgs::msg::Header header{};
  header.stamp.sec = tensor_view.GetTimestampSeconds();
  header.stamp.nanosec = tensor_view.GetTimestampNanoseconds();
  header.frame_id = tensor_view.GetFrameId();

  // Convert NitrosCameraInfo to standard ROS CameraInfo
  sensor_msgs::msg::CameraInfo ros_camera_info;
  try {
    rclcpp::TypeAdapter<nvidia::isaac_ros::nitros::NitrosCameraInfo, sensor_msgs::msg::CameraInfo>
    ::convert_to_ros_message(*camera_info_msg, ros_camera_info);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(this->get_logger(),
    "Failed to convert NitrosCameraInfo to ROS CameraInfo: %s", e.what());
    return;
  }

  // Allocate GPU buffer and copy tensor data
  void * gpu_data;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&gpu_data, tensor.GetTensorSize(), stream_),
    "Failed to allocate GPU buffer for disparity tensor");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      gpu_data, tensor.GetBuffer(),
      tensor.GetTensorSize(), cudaMemcpyDefault, stream_),
    "Failed to copy disparity tensor to GPU");

  // Filter disparity map in-place on GPU
  nvidia::isaac_ros::foundationstereo::FilterDisparity(
    static_cast<float *>(gpu_data),
    static_cast<uint32_t>(width), static_cast<uint32_t>(height),
    static_cast<float>(min_disparity_), static_cast<float>(max_disparity_),
    stream_);

  // Post-launch CUDA error check to surface kernel launch errors
  CHECK_CUDA_ERROR(cudaGetLastError(), "CUDA error after FilterDisparity kernel");

  // Create NitrosDisparityImage using the builder pattern
  auto nitros_disparity_image = nvidia::isaac_ros::nitros::NitrosDisparityImageBuilder()
    .WithHeader(header)
    .WithDimensions(height, width)
    .WithGpuData(gpu_data)
    .WithReleaseCallback([gpu_data, stream = stream_]() {
      cudaFreeAsync(gpu_data, stream);
    })
    .WithDisparityParameters(
      ros_camera_info.p[0],  // focal_length_x from projection matrix
      -ros_camera_info.p[3] / ros_camera_info.p[0],  // baseline from projection matrix
      min_disparity_,
      max_disparity_)
    .Build();

  // Publish Nitros disparity image
  nitros_pub_->publish(nitros_disparity_image);
}

FoundationStereoDecoderNode::~FoundationStereoDecoderNode()
{
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_), "Failed to destroy CUDA stream");
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode)
