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

#include <gtest/gtest.h>
#include "ess_disparity_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception

class ESSDisparityNodeTestSuite : public ::testing::Test
{
protected:
  void SetUp() {rclcpp::init(0, nullptr);}
  void TearDown() {(void)rclcpp::shutdown();}
};


void test_empty_engine_path()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "engine_file_path:=''",
  });
  try {
    nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode ess_disparity_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Empty engine_file_path") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}

void test_image_type()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "engine_file_path:='isaac_ros_dev.engine'",
    "-p", "image_type:='GBR_U8'",
  });
  try {
    nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode ess_disparity_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Only support image_type RGB_U8 and BGR_U8") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}


TEST_F(ESSDisparityNodeTestSuite, test_empty_engine_path)
{
  EXPECT_EXIT(test_empty_engine_path(), testing::ExitedWithCode(1), "");
}

TEST_F(ESSDisparityNodeTestSuite, test_image_type)
{
  EXPECT_EXIT(test_image_type(), testing::ExitedWithCode(1), "");
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  return RUN_ALL_TESTS();
}
