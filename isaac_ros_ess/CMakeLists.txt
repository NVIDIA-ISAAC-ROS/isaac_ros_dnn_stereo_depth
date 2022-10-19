# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.8)
project(isaac_ros_ess LANGUAGES C CXX)


# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


# isaac_ros_ess_node
ament_auto_add_library(isaac_ros_ess_node SHARED src/ess_disparity_node.cpp)
target_compile_definitions(isaac_ros_ess_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(isaac_ros_ess_node)
rclcpp_components_register_nodes(isaac_ros_ess_node "nvidia::isaac_ros::dnn_stereo_disparity::ESSDisparityNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_disparity::ESSDisparityNode;$<TARGET_FILE:isaac_ros_ess_node>\n")

# Install visualizer python scripts

install(PROGRAMS
  scripts/isaac_ros_ess_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install package executable
install(TARGETS isaac_ros_ess_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_ess_test.py TIMEOUT "300")
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
