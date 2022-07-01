/**
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

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
