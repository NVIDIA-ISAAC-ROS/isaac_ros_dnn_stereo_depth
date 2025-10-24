# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_ess LANGUAGES C CXX)

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(${ARCHITECTURE} STREQUAL "aarch64")
    add_compile_options(-Wall -Wextra -Wpedantic -DROS_ARCH_AARCH64)
  elseif(${ARCHITECTURE} STREQUAL "x86_64")
    add_compile_options(-Wall -Wextra -Wpedantic -DROS_ARCH_X86_64)
  endif()
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# isaac_ros_ess_node
ament_auto_add_library(ess_disparity_node SHARED src/ess_disparity_node.cpp)
rclcpp_components_register_nodes(ess_disparity_node "nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode;$<TARGET_FILE:ess_disparity_node>\n")
set_target_properties(ess_disparity_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install visualizer python scripts
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/generate_engine.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/isaac_ros_ess_visualizer.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/isaac_ros_ess_depth_validation.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Force use of ROS2 vendor googletest to avoid version conflicts
  set(CMAKE_IGNORE_PATH "/usr/src/googletest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gtest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gmock" ${CMAKE_IGNORE_PATH})

  # Explicitly use ROS2 vendor googletest/gmock
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gtest_vendor/include)
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gmock_vendor/include)

  # Gtest for disparity node
  ament_add_gtest(ess_disparity_node_test test/ess_disparity_node_test.cpp)
  target_link_libraries(ess_disparity_node_test ess_disparity_node)
  target_include_directories(ess_disparity_node_test PUBLIC include/isaac_ros_ess/)
  ament_target_dependencies(ess_disparity_node_test rclcpp)
  ament_target_dependencies(ess_disparity_node_test isaac_ros_nitros)

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_ess_test.py TIMEOUT "300")
  add_launch_test(test/isaac_ros_ess_test_1_16HD_model.py TIMEOUT "300")
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch config)
