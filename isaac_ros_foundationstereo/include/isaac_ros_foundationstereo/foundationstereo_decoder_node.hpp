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

#ifndef ISAAC_ROS_FOUNDATIONSTEREO__FOUNDATIONSTEREO_DECODER_NODE_HPP_
#define ISAAC_ROS_FOUNDATIONSTEREO__FOUNDATIONSTEREO_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image_builder.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

class FoundationStereoDecoderNode : public rclcpp::Node
{
public:
  explicit FoundationStereoDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~FoundationStereoDecoderNode();

private:
  void SynchronizedCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg);
  void UnsynchronizedCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg);

  // Helper function to process tensor and camera info (common logic for both callbacks)
  void ProcessTensorAndCameraInfo(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg);

  // QOS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Message filter subscribers for synchronization
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosTensorListView> tensor_nitros_sub_;
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosCameraInfo> camera_info_sub_;

  // Message filter synchronizer
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosTensorList,
    nvidia::isaac_ros::nitros::NitrosCameraInfo>;
  message_filters::Synchronizer<ExactPolicy> sync_;

  // Publisher for output NitrosDisparityImage messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosDisparityImage>> nitros_pub_;

  // Tensor names and parameters
  std::string disparity_tensor_name_{};
  double min_disparity_{};
  double max_disparity_{};

  // Constants
  static constexpr int height_dim_{2};
  static constexpr int width_dim_{3};

  // CUDA stream for GPU operations
  cudaStream_t stream_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONSTEREO__FOUNDATIONSTEREO_DECODER_NODE_HPP_
