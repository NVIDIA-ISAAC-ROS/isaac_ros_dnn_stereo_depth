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

#include "isaac_ros_ess/ess_disparity_node.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <iostream>

#include "isaac_ros_common/qos.hpp"

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_LEFT_COMPONENT_KEY[] = "sync/left_image_receiver";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_rgb8";
constexpr char INPUT_LEFT_TOPIC_NAME[] = "left/image_rect";

constexpr char INPUT_RIGHT_COMPONENT_KEY[] = "sync/right_image_receiver";
constexpr char INPUT_RIGHT_TOPIC_NAME[] = "right/image_rect";

constexpr char OUTPUT_COMPONENT_KEY[] = "image_sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char OUTPUT_TOPIC_NAME[] = "disparity";

constexpr char OUTPUT_CAM_COMPONENT_KEY[] = "camera_info_sink/sink";
constexpr char OUTPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char OUTPUT_CAM_TOPIC_NAME[] = "camera_info";

constexpr char INPUT_LEFT_CAM_COMPONENT_KEY[] = "sync/left_cam_receiver";
constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_LEFT_CAMERA_TOPIC_NAME[] = "left/camera_info";

constexpr char INPUT_RIGHT_CAM_COMPONENT_KEY[] = "sync/right_cam_receiver";
constexpr char INPUT_RIGHT_CAMERA_TOPIC_NAME[] = "right/camera_info";

constexpr char APP_YAML_FILENAME[] = "config/ess_inference.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_ess";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"gxf_isaac_video_buffer_utils", "gxf/lib/libgxf_isaac_video_buffer_utils.so"},
  {"gxf_isaac_messages", "gxf/lib/libgxf_isaac_messages.so"},
  {"gxf_isaac_tensorops", "gxf/lib/libgxf_isaac_tensorops.so"},
  {"gxf_isaac_sgm", "gxf/lib/libgxf_isaac_sgm.so"},
  {"gxf_isaac_messages_throttler", "gxf/lib/libgxf_isaac_messages_throttler.so"},
  {"gxf_isaac_ess", "gxf/lib/libgxf_isaac_ess.so"},
  {"gxf_isaac_message_compositor", "gxf/lib/libgxf_isaac_message_compositor.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_ess",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_LEFT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_LEFT_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_RIGHT_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_LEFT_COMPONENT_KEY,
    }
  },
  {INPUT_LEFT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_LEFT_CAMERA_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_RIGHT_CAMERA_TOPIC_NAME,
    }
  },
  {OUTPUT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_CAM_INFO_FORMAT,
      .topic_name = OUTPUT_CAM_TOPIC_NAME,
      .frame_id_source_key = INPUT_LEFT_COMPONENT_KEY,
    }
  }
};
#pragma GCC diagnostic pop

ESSDisparityNode::ESSDisparityNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  image_type_(declare_parameter<std::string>("image_type", "")),
  input_layer_width_(declare_parameter<int>("input_layer_width", 960)),
  input_layer_height_(declare_parameter<int>("input_layer_height", 576)),
  model_input_type_(declare_parameter<std::string>("model_input_type", "RGB_U8")),
  onnx_file_path_(declare_parameter<std::string>("onnx_file_path", "")),
  engine_file_path_(declare_parameter<std::string>("engine_file_path", "")),
  input_layers_name_(declare_parameter<std::vector<std::string>>(
      "input_layers_name", {"input_left", "input_right"})),
  output_layers_name_(declare_parameter<std::vector<std::string>>(
      "output_layers_name", {"output_left", "output_conf"})),
  threshold_(declare_parameter<float>("threshold", 0.4)),
  throttler_skip_(declare_parameter<int>("throttler_skip", 0))
{
  RCLCPP_DEBUG(get_logger(), "[ESSDisparityNode] Initializing ESSDisparityNode.");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  rclcpp::QoS input_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos");
  rclcpp::QoS output_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos");
  for (auto & config : config_map_) {
    if (config.second.topic_name == INPUT_LEFT_TOPIC_NAME ||
      config.second.topic_name == INPUT_RIGHT_TOPIC_NAME ||
      config.second.topic_name == INPUT_LEFT_CAMERA_TOPIC_NAME ||
      config.second.topic_name == INPUT_RIGHT_CAMERA_TOPIC_NAME)
    {
      config.second.qos = input_qos_;
    } else {
      config.second.qos = output_qos_;
    }
  }

  if (engine_file_path_.empty()) {
    throw std::invalid_argument("[ESSDisparityNode] Empty engine_file_path");
  }

  if (!image_type_.empty()) {
    if (image_type_ != "RGB_U8" && image_type_ != "BGR_U8") {
      RCLCPP_INFO(
        get_logger(), "[ESSDisparityNode] Unsupported image type: %s.", image_type_.c_str());
      throw std::invalid_argument("[ESSDisparityNode] Only support image_type RGB_U8 and BGR_U8.");
    }
    auto nitros_format = image_type_ == "RGB_U8" ? "nitros_image_rgb8" : "nitros_image_bgr8";
    config_map_[INPUT_LEFT_COMPONENT_KEY].compatible_data_format = nitros_format;
    config_map_[INPUT_LEFT_COMPONENT_KEY].use_compatible_format_only = true;
    config_map_[INPUT_RIGHT_COMPONENT_KEY].compatible_data_format = nitros_format;
    config_map_[INPUT_RIGHT_COMPONENT_KEY].use_compatible_format_only = true;
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void ESSDisparityNode::preLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[ESSDisparityNode] preLoadGraphCallback().");

#ifdef ROS_ARCH_AARCH64
  constexpr char plugin_prefix[] = "/plugins/aarch64";
#else
  constexpr char plugin_prefix[] = "/plugins/x86_64";
#endif

  std::filesystem::path engine_file(engine_file_path_);
  std::string plugin_path = engine_file.parent_path().generic_string() +
    std::string(plugin_prefix) + "/ess_plugins.so";

  if (std::filesystem::exists(plugin_path)) {
    NitrosNode::preLoadGraphSetParameter(
      "ess", "nvidia::isaac::ESSInference", "tensorrt_plugin", plugin_path);
    RCLCPP_INFO(
      get_logger(), "[ESSDisparityNode] Setting tensorrt_plugin: %s.", plugin_path.c_str());
  } else {
    RCLCPP_WARN(
      get_logger(), "[ESSDisparityNode] TensorRT plugin not found: %s. "
      "This is an issue if running ESS >=4.0.", plugin_path.c_str());
  }
}

void ESSDisparityNode::postLoadGraphCallback()
{
  const uint64_t block_size = sizeof(uint32_t) * input_layer_width_ * input_layer_height_;

  // Forward ESSDisparityNode parameters
  getNitrosContext().setParameterInt32(
    "ess", "nvidia::isaac::ESSInference", "input_layer_width", input_layer_width_);
  getNitrosContext().setParameterInt32(
    "ess", "nvidia::isaac::ESSInference", "input_layer_height", input_layer_height_);
  getNitrosContext().setParameterStr(
    "ess", "nvidia::isaac::ESSInference", "model_input_type", model_input_type_);
  getNitrosContext().setParameterStr(
    "ess", "nvidia::isaac::ESSInference", "onnx_file_path", onnx_file_path_);
  getNitrosContext().setParameterStr(
    "ess", "nvidia::isaac::ESSInference", "engine_file_path", engine_file_path_);
  getNitrosContext().setParameter1DStrVector(
    "ess", "nvidia::isaac::ESSInference", "input_layers_name", input_layers_name_);
  getNitrosContext().setParameter1DStrVector(
    "ess", "nvidia::isaac::ESSInference", "output_layers_name", output_layers_name_);
  getNitrosContext().setParameterFloat32(
    "disparity_thresholder", "nvidia::isaac::VideoBufferThresholder", "threshold", threshold_);
  getNitrosContext().setParameterUInt64(
    "ess", "nvidia::gxf::BlockMemoryPool", "block_size", block_size);
  getNitrosContext().setParameterUInt64(
    "disparity_thresholder", "nvidia::gxf::BlockMemoryPool", "block_size", block_size);
  getNitrosContext().setParameterInt32(
    "throttler", "throttler1", "nvidia::isaac::MultiChannelsThrottler", "skip", throttler_skip_);
  getNitrosContext().setParameterInt32(
    "throttler", "throttler2", "nvidia::isaac::MultiChannelsThrottler", "skip", throttler_skip_);

  RCLCPP_INFO(
    get_logger(), "[ESSDisparityNode] Setting engine_file_path: %s.", engine_file_path_.c_str());
  RCLCPP_INFO(
    get_logger(),
    "[ESSDisparityNode] postLoadGraphCallback() with image [%d x %d]", input_layer_height_,
    input_layer_width_);
  RCLCPP_INFO(
    get_logger(),
    "[ESSDisparityNode] postLoadGraphCallback() block_size = %ld.",
    block_size);
}

ESSDisparityNode::~ESSDisparityNode() {}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode)
