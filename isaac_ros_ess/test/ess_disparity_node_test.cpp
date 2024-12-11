// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock.h>
#include "ess_disparity_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception


TEST(ess_disparity_node_test, test_empty_engine_path)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("engine_file_path", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode ess_disparity_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty engine_file_path"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(ess_disparity_node_test, test_image_type)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("engine_file_path", "ess.engine");
  options.append_parameter_override("image_type", "invalid");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode ess_disparity_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Only support image_type RGB_U8 and BGR_U8."));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
