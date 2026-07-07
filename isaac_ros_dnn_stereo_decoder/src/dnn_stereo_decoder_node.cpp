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

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

DNNStereoDecoderNode::DNNStereoDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("dnn_stereo_decoder_node", options),
  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  cache_camera_info_{declare_parameter<bool>("cache_camera_info", false)},
  tensor_nitros_sub_{},
  camera_info_sub_{},
  sync_{ExactPolicy{3}, tensor_nitros_sub_, camera_info_sub_},
  disparity_tensor_name_{declare_parameter<std::string>(
      "disparity_tensor_name",
      "disparity")},
  confidence_tensor_name_{declare_parameter<std::string>(
      "confidence_tensor_name",
      "")},
  min_disparity_{declare_parameter<double>("min_disparity", 0.0)},
  max_disparity_{declare_parameter<double>("max_disparity", 10000.0)},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.0)},
  reusable_buffer_enable_{declare_parameter<bool>("reusable_buffer_enable", true)},
  reusable_buffer_count_{declare_parameter<int>("reusable_buffer_count", 2)},
  reusable_buffer_width_{declare_parameter<int>("reusable_buffer_width", 576)},
  reusable_buffer_height_{declare_parameter<int>("reusable_buffer_height", 960)}
{
  CHECK_CUDA_ERROR(
    nvidia::isaac_ros::common::initNamedCudaStream(stream_, "dnn_stereo_decoder_node"),
    "Failed to initialize CUDA stream");

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  nitros_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosDisparityImage>(
    "disparity", output_qos_, pub_options);

  // Validate confidence_threshold_ is within range [0, 1]
  if (confidence_threshold_ < 0.0 || confidence_threshold_ > 1.0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "confidence_threshold must be in range [0.0, 1.0], got: %f",
      confidence_threshold_);
    throw std::invalid_argument("confidence_threshold out of range [0.0, 1.0]");
  }

  // Optional reusable buffer pool at startup if user provided fixed dims
  if (reusable_buffer_enable_ && reusable_buffer_width_ > 0 && reusable_buffer_height_ > 0) {
    AllocateReusableBufferPool(
      static_cast<uint32_t>(reusable_buffer_width_),
      static_cast<uint32_t>(reusable_buffer_height_),
      reusable_buffer_count_ > 0 ? reusable_buffer_count_ : 2);
  }

  if (cache_camera_info_) {
    // Caching mode: Subscribe to topics independently without synchronization
    // Camera info is cached and every tensor message is processed immediately
    RCLCPP_INFO(this->get_logger(),
      "Camera info caching enabled - processing every tensor message with cached camera info");

    // Create independent subscribers for caching mode
    camera_info_sub_cached_mode_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "right/camera_info", input_qos_,
      std::bind(&DNNStereoDecoderNode::CameraInfoCallback, this, std::placeholders::_1));

    tensor_sub_cached_mode_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
      "tensor_sub", input_qos_,
      std::bind(&DNNStereoDecoderNode::TensorCallback, this, std::placeholders::_1));
  } else {
    // Synchronization mode: Use exact time synchronization
    // Only process synchronized messages, discard unsynchronized with warning
    RCLCPP_INFO(this->get_logger(),
      "Using exact time synchronization - unsynchronized messages will be discarded");

    // Subscribe to topics using message filters
    tensor_nitros_sub_.subscribe(this, "tensor_sub");
    camera_info_sub_.subscribe(this, "right/camera_info");

    // Register synchronized callback
    sync_.registerCallback(
      std::bind(
        &DNNStereoDecoderNode::SynchronizedCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    // Register drop callback for unsynchronized messages
    sync_.getPolicy()->registerDropCallback(
      std::bind(
        &DNNStereoDecoderNode::UnsynchronizedCallback, this,
        std::placeholders::_1, std::placeholders::_2));
  }
}

void DNNStereoDecoderNode::SynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Processing synchronized tensor and camera info pair!");

  if (static_cast<int32_t>(tensor_msg->get_timestamp_sec()) !=
    camera_info_msg->header.stamp.sec ||
    tensor_msg->get_timestamp_nsec() != camera_info_msg->header.stamp.nanosec)
  {
    RCLCPP_WARN(this->get_logger(), "Both messages received, but timestamps didn't match!");
    return;
  }

  ProcessTensorAndCameraInfo(tensor_msg, camera_info_msg);
}

void DNNStereoDecoderNode::UnsynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // In synchronization mode, unsynchronized messages are discarded with a warning
  if (!tensor_msg && !camera_info_msg) {
    RCLCPP_WARN(this->get_logger(),
      "Both tensor and camera info messages missing - skipping the frame");
  } else if (!tensor_msg) {
    RCLCPP_WARN(this->get_logger(),
      "Tensor message missing - skipping the frame");
  } else if (!camera_info_msg) {
    RCLCPP_WARN(this->get_logger(),
      "Camera info message missing - skipping the frame");
  } else {
    RCLCPP_WARN(this->get_logger(),
      "Tensor and camera info pair dropped due to timestamp mismatch - "
      "consider enabling 'cache_camera_info' parameter to process all tensor messages");
  }
  // Message is discarded - not processed
}

void DNNStereoDecoderNode::CameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // Cache the latest camera info for use with incoming tensor messages
  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  stored_camera_info_ = camera_info_msg;
  RCLCPP_DEBUG(this->get_logger(), "Camera info cached");
}

void DNNStereoDecoderNode::TensorCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg)
{
  // Get the cached camera info
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_info = stored_camera_info_;
  }

  if (!camera_info) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Tensor received but no camera info cached yet - waiting for camera info");
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "Processing tensor with cached camera info");
  ProcessTensorAndCameraInfo(tensor_msg, camera_info);
}

void DNNStereoDecoderNode::ProcessTensorAndCameraInfo(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // Guard against degenerate projection matrix that would make baseline undefined
  if (std::abs(camera_info_msg->p[0]) <= std::numeric_limits<double>::epsilon()) {
    RCLCPP_WARN(this->get_logger(),
      "Camera info focal_length_x (p[0]=%f) is zero/near-zero; skipping frame",
      camera_info_msg->p[0]);
    return;
  }

  // Get tensor dimensions
  auto tensor_ptr = tensor_msg->get_tensor_by_name(disparity_tensor_name_);
  if (!tensor_ptr) {
    RCLCPP_ERROR(this->get_logger(),
      "Tensor '%s' not found in tensor list", disparity_tensor_name_.c_str());
    return;
  }
  const auto & tensor = *tensor_ptr;
  const uint32_t rank = tensor.GetRank();
  const int dynamic_height_dim = ComputeHeightDim(rank);
  const int dynamic_width_dim = ComputeWidthDim(rank);
  int height = tensor.GetShape().dims()[dynamic_height_dim];
  int width = tensor.GetShape().dims()[dynamic_width_dim];

  // Create header
  std_msgs::msg::Header header{};
  header.stamp.sec = static_cast<int32_t>(tensor_msg->get_timestamp_sec());
  header.stamp.nanosec = tensor_msg->get_timestamp_nsec();
  header.frame_id = tensor_msg->get_frame_id();

  // Allocate GPU buffer and copy tensor data
  void * gpu_data = nullptr;
  const size_t tensor_bytes = tensor.GetTensorSize();

  if (reusable_buffer_enable_) {
    const uint32_t req_width = static_cast<uint32_t>(width);
    const uint32_t req_height = static_cast<uint32_t>(height);
    const size_t req_bytes = ComputeRequiredBytes(req_width, req_height);

    // Initialize or resize pool if needed
    if (reusable_buffer_size_bytes_ == 0) {
      AllocateReusableBufferPool(req_width, req_height, reusable_buffer_count_);
    } else if (req_bytes != reusable_buffer_size_bytes_) {
      RCLCPP_WARN(this->get_logger(),
          "Disparity tensor size changed (%zu -> %zu). Reinitializing reusable buffer pool.",
          reusable_buffer_size_bytes_, req_bytes);
      FreeReusableBufferPool();
      AllocateReusableBufferPool(req_width, req_height,
          reusable_buffer_count_ > 0 ? reusable_buffer_count_ : 2);
    }

    if (reusable_buffer_size_bytes_ == req_bytes) {
      gpu_data = AcquireReusableBuffer();
      if (gpu_data == nullptr) {
        RCLCPP_ERROR(this->get_logger(),
          "No free reusable buffers available. All %d buffers are in use. "
          "Increase 'reusable_buffer_count' parameter to handle higher message throughput.",
          reusable_buffer_count_);
        return;
      }
    }
  } else {
    // Per-frame allocation when reusable buffer is disabled
    CHECK_CUDA_ERROR(
      cudaMallocAsync(&gpu_data, tensor_bytes, stream_),
      "Failed to allocate GPU buffer for disparity tensor");
  }

  if (gpu_data == nullptr) {
    RCLCPP_ERROR(this->get_logger(),
      "Failed to acquire GPU buffer - buffer allocation failed");
    return;
  }

  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      gpu_data, tensor.GetBuffer(stream_),
      tensor_bytes, cudaMemcpyDefault, stream_),
    "Failed to copy disparity tensor to GPU");

  // Apply confidence masking if confidence map exists and threshold > 0.
  // Pixels with confidence below the threshold are discarded in the disparity map.
  // Primarily used by the ESS model.
  if (confidence_threshold_ > 0.0) {
    std::string conf_name = confidence_tensor_name_;
    if (!conf_name.empty()) {
      auto conf_tensor_ptr = tensor_msg->get_tensor_by_name(conf_name);
      if (!conf_tensor_ptr) {
        RCLCPP_WARN(this->get_logger(),
          "Confidence tensor '%s' not found; skipping confidence masking",
          conf_name.c_str());
      } else {
        const auto & conf_tensor = *conf_tensor_ptr;
        const uint32_t conf_rank = conf_tensor.GetRank();
        const int conf_height_dim = ComputeHeightDim(conf_rank);
        const int conf_width_dim = ComputeWidthDim(conf_rank);
        int conf_height = conf_tensor.GetShape().dims()[conf_height_dim];
        int conf_width = conf_tensor.GetShape().dims()[conf_width_dim];
        if (conf_height == height && conf_width == width &&
          conf_tensor.bytes_per_element() == sizeof(float))
        {
          const void * conf_device_ptr = conf_tensor.GetBuffer(stream_);
          const cudaError_t conf_err =
            nvidia::isaac_ros::dnn_stereo_decoder::ApplyConfidenceThreshold(
            static_cast<float *>(gpu_data),
            static_cast<const float *>(conf_device_ptr),
            static_cast<uint32_t>(width), static_cast<uint32_t>(height),
            static_cast<float>(confidence_threshold_), stream_);
          CHECK_CUDA_ERROR(conf_err, "CUDA error after ApplyConfidenceThreshold kernel");
        } else {
          RCLCPP_WARN(this->get_logger(),
            "Confidence tensor dims/type mismatch; skipping confidence masking "
            "(expected %dx%d float)",
            height, width);
        }
      }
    }
  }

  // Filter disparity map in-place on GPU
  const cudaError_t filter_err =
    nvidia::isaac_ros::dnn_stereo_decoder::FilterDisparity(
    static_cast<float *>(gpu_data),
    static_cast<uint32_t>(width), static_cast<uint32_t>(height),
    static_cast<float>(min_disparity_), static_cast<float>(max_disparity_),
    stream_);

  // Post-launch CUDA error check to surface kernel launch errors
  CHECK_CUDA_ERROR(filter_err, "CUDA error after FilterDisparity kernel");

  // Create NitrosDisparityImage using the builder pattern
  nvidia::isaac_ros::nitros::NitrosDisparityImageBuilder builder;
  builder.WithHeader(header)
  .WithDimensions(height, width)
  .WithGpuData(gpu_data)
  .WithDisparityParameters(
    camera_info_msg->p[0],  // focal_length_x from projection matrix
    -camera_info_msg->p[3] / camera_info_msg->p[0],  // baseline from projection matrix
    min_disparity_,
    max_disparity_);

  // Provide appropriate release callback depending on allocation mode
  if (reusable_buffer_enable_ && reusable_buffer_size_bytes_ != 0) {
    // Find the entry's in_use flag to release via captured shared_ptr
    std::shared_ptr<std::atomic<bool>> in_use_flag{};
    for (const auto & entry : reusable_buffers_) {
      if (entry.ptr == gpu_data) {
        in_use_flag = entry.in_use;
        break;
      }
    }
    if (in_use_flag) {
      builder.WithReleaseCallback([in_use_flag]() {
          DNNStereoDecoderNode::ReleaseReusableBuffer(in_use_flag);
      });
    } else {
      // Fallback: if pointer didn't come from pool, free it
      builder.WithReleaseCallback([gpu_data, stream = stream_]() {
          cudaFreeAsync(gpu_data, stream);
      });
    }
  } else {
    builder.WithReleaseCallback([gpu_data, stream = stream_]() {
        cudaFreeAsync(gpu_data, stream);
    });
  }

  // Ensure all GPU operations on stream_ have completed before building/publishing
  CHECK_CUDA_ERROR(
    cudaStreamSynchronize(stream_),
    "Failed to synchronize CUDA stream before publishing");

  auto nitros_disparity_image = builder.Build();

  // Publish Nitros disparity image
  nitros_pub_->publish(std::move(nitros_disparity_image));
}

DNNStereoDecoderNode::~DNNStereoDecoderNode()
{
  // Free preallocated buffers if any
  FreeReusableBufferPool();
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_), "Failed to destroy CUDA stream");
}

void DNNStereoDecoderNode::AllocateReusableBufferPool(uint32_t width, uint32_t height, int count)
{
  if (count <= 0) {
    count = 2;
  }
  reusable_buffers_.clear();
  reusable_buffers_.reserve(static_cast<size_t>(count));
  reusable_buffer_size_bytes_ = ComputeRequiredBytes(width, height);
  reusable_buffer_width_runtime_ = width;
  reusable_buffer_height_runtime_ = height;

  for (int i = 0; i < count; ++i) {
    void * ptr = nullptr;
    CHECK_CUDA_ERROR(
      cudaMallocAsync(&ptr, reusable_buffer_size_bytes_, stream_),
      "Failed to allocate reusable CUDA buffer %d of %zu bytes", i, reusable_buffer_size_bytes_);
    ReusableBufferEntry entry;
    entry.ptr = ptr;
    entry.in_use = std::make_shared<std::atomic<bool>>(false);
    reusable_buffers_.emplace_back(std::move(entry));
  }
  RCLCPP_INFO(this->get_logger(),
    "Initialized %d reusable CUDA buffers (%ux%u, %zu bytes each)",
    count, width, height, reusable_buffer_size_bytes_);
}

void DNNStereoDecoderNode::FreeReusableBufferPool()
{
  if (reusable_buffer_size_bytes_ == 0 || reusable_buffers_.empty()) {
    reusable_buffers_.clear();
    reusable_buffer_size_bytes_ = 0;
    reusable_buffer_width_runtime_ = 0;
    reusable_buffer_height_runtime_ = 0;
    return;
  }
  for (auto & entry : reusable_buffers_) {
    if (entry.ptr != nullptr) {
      // Free asynchronously on node stream
      CHECK_CUDA_ERROR(cudaFreeAsync(entry.ptr, stream_), "Failed to free preallocated buffer");
      entry.ptr = nullptr;
    }
  }
  reusable_buffers_.clear();
  reusable_buffer_size_bytes_ = 0;
  reusable_buffer_width_runtime_ = 0;
  reusable_buffer_height_runtime_ = 0;
}

void * DNNStereoDecoderNode::AcquireReusableBuffer()
{
  for (auto & entry : reusable_buffers_) {
    bool expected = false;
    if (entry.in_use->compare_exchange_strong(expected, true)) {
      return entry.ptr;
    }
  }
  return nullptr;
}

void DNNStereoDecoderNode::ReleaseReusableBuffer(
  const std::shared_ptr<std::atomic<bool>> & in_use_flag)
{
  if (in_use_flag) {
    in_use_flag->store(false);
  }
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode)
