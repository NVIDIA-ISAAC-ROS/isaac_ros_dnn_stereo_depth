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

#include "isaac_ros_ess/ess_disparity_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_disparity
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_LEFT_COMPONENT_KEY[] = "sync/left_image_receiver";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_rgb8";
constexpr char INPUT_LEFT_TOPIC_NAME[] = "left/image_rect";

constexpr char INPUT_RIGHT_COMPONENT_KEY[] = "sync/right_image_receiver";
constexpr char INPUT_RIGHT_TOPIC_NAME[] = "right/image_rect";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char OUTPUT_TOPIC_NAME[] = "disparity";

constexpr char INPUT_LEFT_CAM_COMPONENT_KEY[] = "sync/left_cam_receiver";
constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_LEFT_CAMERA_TOPIC_NAME[] = "left/camera_info";

constexpr char INPUT_RIGHT_CAM_COMPONENT_KEY[] = "sync/right_cam_receiver";
constexpr char INPUT_RIGHT_CAMERA_TOPIC_NAME[] = "right/camera_info";

constexpr char APP_YAML_FILENAME[] = "config/ess_inference.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_ess";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/libgxf_synchronization.so"},
  {"isaac_ros_image_proc", "gxf/lib/image_proc/libgxf_tensorops.so"},
  {"isaac_ros_stereo_image_proc", "gxf/lib/sgm_disparity/libgxf_disparity_extension.so"},
  {"isaac_ros_ess", "gxf/lib/ess/libgxf_cvcore_ess.so"}
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
  engine_file_path_(declare_parameter<std::string>("engine_file_path", "")),
  input_layers_name_(declare_parameter<std::vector<std::string>>(
      "input_layers_name", {"input_left", "input_right"})),
  output_layers_name_(declare_parameter<std::vector<std::string>>(
      "output_layers_name", {"output_left"}))
{
  RCLCPP_DEBUG(get_logger(), "[ESSDisparityNode] Initializing ESSDisparityNode.");

  if (engine_file_path_.empty()) {
    throw std::invalid_argument("[ESSDisparityNode] Empty engine_file_path");
  }

  if (!image_type_.empty()) {
    if (image_type_ != "RGB_U8" && image_type_ != "BGR_U8") {
      RCLCPP_INFO(
        get_logger(), "[ESSDisparityNode] Unspported image type: %s.", image_type_.c_str());
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

void ESSDisparityNode::postLoadGraphCallback()
{
  // Forward ESSDisparityNode parameters
  getNitrosContext().setParameterInt32(
    "ess", "nvidia::cvcore::ESS", "input_layer_width", input_layer_width_);
  getNitrosContext().setParameterInt32(
    "ess", "nvidia::cvcore::ESS", "input_layer_height", input_layer_height_);
  getNitrosContext().setParameterStr(
    "ess", "nvidia::cvcore::ESS", "model_input_type", model_input_type_);
  getNitrosContext().setParameterStr(
    "ess", "nvidia::cvcore::ESS", "engine_file_path", engine_file_path_);
  getNitrosContext().setParameter1DStrVector(
    "ess", "nvidia::cvcore::ESS", "input_layers_name", input_layers_name_);
  getNitrosContext().setParameter1DStrVector(
    "ess", "nvidia::cvcore::ESS", "output_layers_name", output_layers_name_);

  RCLCPP_INFO(
    get_logger(), "[ESSDisparityNode] Setting engine_file_path: %s.", engine_file_path_.c_str());
}

ESSDisparityNode::~ESSDisparityNode() {}

}  // namespace dnn_stereo_disparity
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_disparity::ESSDisparityNode)
