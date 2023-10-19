# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_cvcore_ess LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

# Dependencies
find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(isaac_ros_image_proc REQUIRED)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    multimedia
    serialization
)
find_package(TENSORRT)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# Create extension
add_library(gxf_cvcore_ess SHARED
  extensions/ess/ess.cpp
  extensions/ess/components/ess_inference.cpp
  extensions/ess/components/ess_inference.hpp

  extensions/ess/inference/ESS.cpp
  extensions/ess/inference/ESS.h
  extensions/ess/inference/ESSPostProcess.cpp
  extensions/ess/inference/ESSPreProcess.cpp
)
target_include_directories(gxf_cvcore_ess PRIVATE ${isaac_ros_image_proc_INCLUDE_DIRS})

add_library(corelib STATIC
  # Inferencer (ESS only)
  gems/dnn_inferencer/inferencer/TensorRTInferencer.cpp
  gems/dnn_inferencer/inferencer/TensorRTUtils.h
  gems/dnn_inferencer/inferencer/TensorRTUtils.cpp
  gems/dnn_inferencer/inferencer/TensorRTInferencer.h
  gems/dnn_inferencer/inferencer/Inferencer.cpp
  gems/dnn_inferencer/inferencer/Errors.cpp
)
target_include_directories(corelib PRIVATE ${isaac_ros_image_proc_INCLUDE_DIRS})

target_link_libraries(corelib PUBLIC
    GXF::core
    GXF::cuda
    GXF::multimedia
    CUDA::cudart
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppidei
    CUDA::nppif
    CUDA::nppig
    CUDA::nppisu
    TENSORRT::nvinfer
    yaml-cpp
)

target_compile_options(gxf_cvcore_ess PUBLIC -fPIC)

target_link_libraries(gxf_cvcore_ess
    corelib
    isaac_ros_image_proc::gxf_tensorops
)