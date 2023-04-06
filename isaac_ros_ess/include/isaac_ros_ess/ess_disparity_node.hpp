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

#ifndef ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_
#define ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_

#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_disparity
{

class ESSDisparityNode : public nitros::NitrosNode
{
public:
  explicit ESSDisparityNode(const rclcpp::NodeOptions &);

  ~ESSDisparityNode();

  ESSDisparityNode(const ESSDisparityNode &) = delete;

  ESSDisparityNode & operator=(const ESSDisparityNode &) = delete;

  // The callback for submitting parameters to the node's graph
  void postLoadGraphCallback() override;

private:
  const std::string image_type_;
  const int input_layer_width_;
  const int input_layer_height_;
  const std::string model_input_type_;
  const std::string engine_file_path_;
  const std::vector<std::string> input_layers_name_;
  const std::vector<std::string> output_layers_name_;
};

}  // namespace dnn_stereo_disparity
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_ESS__ESS_DISPARITY_NODE_HPP_
