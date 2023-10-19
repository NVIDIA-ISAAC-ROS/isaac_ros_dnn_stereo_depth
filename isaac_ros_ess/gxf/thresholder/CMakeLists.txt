# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_thresholder LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()
enable_language(CUDA)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    multimedia
    serialization
    isaac_messages
)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# Create extension
add_library(gxf_thresholder SHARED
  extensions/video_buffer_utils/video_buffer_thresholder_extension.cpp
  extensions/video_buffer_utils/components/video_buffer_thresholder.cpp
  extensions/video_buffer_utils/components/video_buffer_thresholder.hpp
  extensions/video_buffer_utils/gems/video_buffer_thresholder.cu
  extensions/video_buffer_utils/gems/video_buffer_thresholder.cu.hpp
)

target_link_libraries(gxf_thresholder
  PRIVATE
    Eigen3::Eigen
    GXF::isaac_messages
  PUBLIC
    GXF::core
    GXF::cuda
    GXF::multimedia
    CUDA::cudart
    yaml-cpp
)

target_compile_options(gxf_thresholder PUBLIC -fPIC)
