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
project(gxf_isaac_ess LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(TENSORRT)

# Create extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/ess/ess.cpp
  gxf/extensions/ess/components/ess_inference.cpp
  gxf/extensions/ess/components/ess_inference.hpp

  gxf/extensions/ess/inference/ESS.cpp
  gxf/extensions/ess/inference/ESS.h
  gxf/extensions/ess/inference/ESSPostProcess.cpp
  gxf/extensions/ess/inference/ESSPreProcess.cpp

  # Inferencer (ESS only)
  gxf/gems/dnn_inferencer/inferencer/TensorRTInferencer.cpp
  gxf/gems/dnn_inferencer/inferencer/TensorRTUtils.h
  gxf/gems/dnn_inferencer/inferencer/TensorRTUtils.cpp
  gxf/gems/dnn_inferencer/inferencer/TensorRTInferencer.h
  gxf/gems/dnn_inferencer/inferencer/Inferencer.cpp
  gxf/gems/dnn_inferencer/inferencer/Errors.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf")

target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  TENSORRT::nvinfer
  yaml-cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)

ament_auto_package(INSTALL_TO_SHARE)
