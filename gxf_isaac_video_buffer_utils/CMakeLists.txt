# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_isaac_video_buffer_utils LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()
enable_language(CUDA)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit)
find_package(yaml-cpp)

# Create extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/video_buffer_utils/video_buffer_thresholder_extension.cpp
  gxf/extensions/video_buffer_utils/components/video_buffer_thresholder.cpp
  gxf/extensions/video_buffer_utils/components/video_buffer_thresholder.hpp
  gxf/extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.cpp
  gxf/extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.hpp
)
# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  gxf/extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.cpp
  gxf/extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.hpp
  PROPERTIES LANGUAGE CUDA
)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf")
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  CUDA::cudart
  yaml-cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_compile_options(${PROJECT_NAME} PUBLIC -fPIC)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE)
