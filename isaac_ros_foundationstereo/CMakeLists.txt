# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_foundationstereo LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED) # Ensure CUDA imported targets (e.g., CUDA::nvtx3 / CUDA::nvToolsExt) are available
include_directories(${OpenCV_INCLUDE_DIRS})
ament_auto_find_build_dependencies()

# NVTX
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    link_libraries("CUDA::nvtx3")
endif()

# FoundationStereoDecoderNode
ament_auto_add_library(foundationstereo_decoder_node SHARED src/foundationstereo_decoder_node.cpp)
target_link_libraries(foundationstereo_decoder_node ${OpenCV_LIBS})
rclcpp_components_register_nodes(foundationstereo_decoder_node "nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode;$<TARGET_FILE:foundationstereo_decoder_node>\n")

# Install visualizer python scripts
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/isaac_ros_foundationstereo_visualizer.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/isaac_ros_foundationstereo_depth_validation.py DESTINATION lib/${PROJECT_NAME})

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
