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

#ifndef ISAAC_ROS_DNN_STEREO_DECODER__DNN_STEREO_DECODER_NODE_HPP_
#define ISAAC_ROS_DNN_STEREO_DECODER__DNN_STEREO_DECODER_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

#include "isaac_ros_dnn_stereo_decoder/filter_disparity.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

using Tensor = tensor_msgs::msg::ExperimentalTensor;
using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

/// Node that converts a disparity tensor output by a DNN into a disparity image message.
/**
 * This node:
 *  - Subscribes to a TensorListMsg disparity tensor and right camera info
 *  - Optionally applies a confidence threshold if a confidence tensor is provided
 *  - Filters invalid/out-of-range disparity values on the GPU
 *  - Publishes a stereo_msgs::DisparityImage with disparity parameters from camera info
 */
class DNNStereoDecoderNode : public rclcpp::Node
{
public:
  /// Construct the node and initialize subscriptions, publisher, and CUDA stream.
  explicit DNNStereoDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  /// Destroy the node and release the CUDA stream.
  ~DNNStereoDecoderNode();

private:
  /// Callback for synchronized tensor and camera info messages (cache_camera_info=false).
  void SynchronizedCallback(
    const TensorList::ConstSharedPtr & tensor_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);
  /// Callback invoked when messages are dropped by the synchronizer (cache_camera_info=false).
  /// Discards the message with a warning log.
  void UnsynchronizedCallback(
    const TensorList::ConstSharedPtr & tensor_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  /// Callback for camera info messages (cache_camera_info=true).
  /// Caches the latest camera info for use with incoming tensor messages.
  void CameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  /// Callback for tensor messages (cache_camera_info=true).
  /// Processes every tensor using the cached camera info.
  void TensorCallback(
    const TensorList::ConstSharedPtr & tensor_msg);

  // Helper function to process tensor and camera info (common logic for both callbacks)
  void ProcessTensorAndCameraInfo(
    const TensorList::ConstSharedPtr & tensor_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  // QOS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Camera info caching mode parameter
  // True: Cache camera info and process every tensor message independently (no synchronization)
  // False: Use exact time synchronization, discard unsynchronized messages with warning
  bool cache_camera_info_{};

  // Cached camera info for caching mode
  sensor_msgs::msg::CameraInfo::ConstSharedPtr stored_camera_info_;
  mutable std::mutex camera_info_mutex_;

  // Message filter subscribers for synchronization mode (cache_camera_info=false)
  message_filters::Subscriber<TensorList> tensor_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  // Message filter synchronizer (cache_camera_info=false)
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    TensorList,
    sensor_msgs::msg::CameraInfo>;
  message_filters::Synchronizer<ExactPolicy> sync_;

  // Separate subscribers for caching mode (cache_camera_info=true)
  rclcpp::Subscription<TensorList>::SharedPtr tensor_sub_cached_mode_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_cached_mode_;

  // Publisher for output DisparityImage messages
  rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_pub_;

  // Tensor names and parameters
  std::string disparity_tensor_name_{};
  std::string confidence_tensor_name_{};
  double min_disparity_{};
  double max_disparity_{};
  double confidence_threshold_{};

  // Compute dims depending on tensor rank (H,W indices are 1,2 for rank-3; else 2,3)
  /// Compute the dimension index for height given the tensor rank.
  static inline int ComputeHeightDim(size_t rank) {return rank == 3 ? 1 : 2;}
  /// Compute the dimension index for width given the tensor rank.
  static inline int ComputeWidthDim(size_t rank) {return rank == 3 ? 2 : 3;}

  // CUDA stream for GPU operations
  cudaStream_t stream_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_STEREO_DECODER__DNN_STEREO_DECODER_NODE_HPP_
