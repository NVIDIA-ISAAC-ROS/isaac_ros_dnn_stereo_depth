# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_dnn_stereo_decoder LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(CUDAToolkit REQUIRED) # Ensure CUDA imported targets (e.g., CUDA::nvtx3 / CUDA::nvToolsExt) are available
ament_auto_find_build_dependencies()

# NVTX
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    link_libraries("CUDA::nvtx3")
endif()

# DNNStereoDecoderNode
ament_auto_add_library(dnn_stereo_decoder_node SHARED
  src/dnn_stereo_decoder_node.cpp
  src/filter_disparity.cu.cpp)
rclcpp_components_register_nodes(dnn_stereo_decoder_node "nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode;$<TARGET_FILE:dnn_stereo_decoder_node>\n")
target_include_directories(dnn_stereo_decoder_node PUBLIC include)
set_target_properties(dnn_stereo_decoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  src/filter_disparity.cu.cpp
  include/isaac_ros_dnn_stereo_decoder/filter_disparity.cu.hpp
  PROPERTIES LANGUAGE CUDA
)

# Install python scripts
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/isaac_ros_dnn_stereo_visualizer.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_dnn_stereo_decoder_test.py TIMEOUT "600")
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
