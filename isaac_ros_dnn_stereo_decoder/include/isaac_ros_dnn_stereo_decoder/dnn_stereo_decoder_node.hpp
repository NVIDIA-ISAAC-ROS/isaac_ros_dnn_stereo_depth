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
#include <string>
#include <vector>
#include <limits>
#include <mutex>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image_builder.hpp"

#include "isaac_ros_dnn_stereo_decoder/filter_disparity.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

/// Node that converts a disparity tensor output by a DNN into a disparity image message.
/**
 * This node:
 *  - Subscribes to a Nitros disparity tensor and right camera info
 *  - Optionally applies a confidence threshold if a confidence tensor is provided
 *  - Filters invalid/out-of-range disparity values on the GPU
 *  - Publishes a NitrosDisparityImage with disparity parameters populated from camera info
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
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);
  /// Callback invoked when messages are dropped by the synchronizer (cache_camera_info=false).
  /// Discards the message with a warning log.
  void UnsynchronizedCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  /// Callback for camera info messages (cache_camera_info=true).
  /// Caches the latest camera info for use with incoming tensor messages.
  void CameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  /// Callback for tensor messages (cache_camera_info=true).
  /// Processes every tensor using the cached camera info.
  void TensorCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg);

  // Helper function to process tensor and camera info (common logic for both callbacks)
  void ProcessTensorAndCameraInfo(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
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
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosTensorList> tensor_nitros_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  // Message filter synchronizer (cache_camera_info=false)
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosTensorList,
    sensor_msgs::msg::CameraInfo>;
  message_filters::Synchronizer<ExactPolicy> sync_;

  // Separate subscribers for caching mode (cache_camera_info=true)
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr
    tensor_sub_cached_mode_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_cached_mode_;

  // Publisher for output NitrosDisparityImage messages
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosDisparityImage>::SharedPtr nitros_pub_;

  // Tensor names and parameters
  std::string disparity_tensor_name_{};
  std::string confidence_tensor_name_{};
  double min_disparity_{};
  double max_disparity_{};
  double confidence_threshold_{};

  // Compute dims depending on tensor rank (H,W indices are 1,2 for rank-3; else 2,3)
  /// Compute the dimension index for height given the tensor rank.
  static inline int ComputeHeightDim(uint32_t rank) {return rank == 3 ? 1 : 2;}
  /// Compute the dimension index for width given the tensor rank.
  static inline int ComputeWidthDim(uint32_t rank) {return rank == 3 ? 2 : 3;}

  // CUDA stream for GPU operations
  cudaStream_t stream_;

  // Optional preallocation settings
  bool reusable_buffer_enable_{};
  int reusable_buffer_count_{};
  int reusable_buffer_width_{};
  int reusable_buffer_height_{};
  bool reusable_buffer_enable_dynamic_{};

  struct ReusableBufferEntry
  {
    void * ptr{nullptr};
    std::shared_ptr<std::atomic<bool>> in_use{std::make_shared<std::atomic<bool>>(false)};
  };

  // Pool of reusable device buffers for output disparity
  std::vector<ReusableBufferEntry> reusable_buffers_;
  size_t reusable_buffer_size_bytes_{0};
  uint32_t reusable_buffer_width_runtime_{0};
  uint32_t reusable_buffer_height_runtime_{0};

  // Pool management
  inline size_t ComputeRequiredBytes(uint32_t width, uint32_t height) const
  {
    return static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float);
  }
  void AllocateReusableBufferPool(uint32_t width, uint32_t height, int count);
  void FreeReusableBufferPool();
  void * AcquireReusableBuffer();
  static void ReleaseReusableBuffer(const std::shared_ptr<std::atomic<bool>> & in_use_flag);
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_STEREO_DECODER__DNN_STEREO_DECODER_NODE_HPP_
