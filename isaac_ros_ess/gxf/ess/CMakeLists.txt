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
include(YamlCpp)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    multimedia
    serialization
)
find_package(TENSORRT)

# Create extension
add_library(gxf_cvcore_ess SHARED
  TensorOps.cpp
  ESS.cpp
  extensions/ess/ESS.hpp

  extensions/tensor_ops/CameraModel.cpp
  extensions/tensor_ops/CameraModel.hpp
  extensions/tensor_ops/ConvertColorFormat.cpp
  extensions/tensor_ops/ConvertColorFormat.hpp
  extensions/tensor_ops/CropAndResize.cpp
  extensions/tensor_ops/CropAndResize.hpp
  extensions/tensor_ops/Frame3D.cpp
  extensions/tensor_ops/Frame3D.hpp
  extensions/tensor_ops/ImageAdapter.cpp
  extensions/tensor_ops/ImageAdapter.hpp
  extensions/tensor_ops/ImageUtils.cpp
  extensions/tensor_ops/ImageUtils.hpp
  extensions/tensor_ops/InterleavedToPlanar.cpp
  extensions/tensor_ops/InterleavedToPlanar.hpp
  extensions/tensor_ops/Normalize.cpp
  extensions/tensor_ops/Normalize.hpp
  extensions/tensor_ops/Reshape.cpp
  extensions/tensor_ops/Reshape.hpp
  extensions/tensor_ops/Resize.cpp
  extensions/tensor_ops/Resize.hpp
  extensions/tensor_ops/TensorOperator.cpp
  extensions/tensor_ops/TensorOperator.hpp
  extensions/tensor_ops/TensorStream.cpp
  extensions/tensor_ops/TensorStream.hpp
  extensions/tensor_ops/Undistort.cpp
  extensions/tensor_ops/Undistort.hpp
  extensions/tensor_ops/detail/ImageAdapterTensorImpl.cpp
  extensions/tensor_ops/detail/ImageAdapterTensorImpl.hpp
  extensions/tensor_ops/detail/ImageAdapterVideoBufferImpl.cpp
  extensions/tensor_ops/detail/ImageAdapterVideoBufferImpl.hpp
)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
target_include_directories(gxf_cvcore_ess PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3dv/include)

add_library(cvcore_ess STATIC
  # Tensorops
  cvcore/src/tensor_ops/ArithmeticOperations.cpp
  cvcore/src/tensor_ops/BBoxUtils.cpp
  cvcore/src/tensor_ops/ColorConversions.cpp
  cvcore/src/tensor_ops/DBScan.cpp
  cvcore/src/tensor_ops/Errors.cpp
  cvcore/src/tensor_ops/Filters.cpp
  cvcore/src/tensor_ops/Filters.h
  cvcore/src/tensor_ops/FusedOperations.cpp
  cvcore/src/tensor_ops/GeometryTransforms.cpp
  cvcore/src/tensor_ops/IImageWarp.cpp
  cvcore/src/tensor_ops/NppUtils.cpp
  cvcore/src/tensor_ops/NppUtils.h
  cvcore/src/tensor_ops/OneEuroFilter.cpp
  cvcore/src/tensor_ops/TensorOperators.cpp

  # Core
  cvcore/src/core/cvcore/Array.cpp
  cvcore/src/core/cvcore/Dummy.cu
  cvcore/src/core/cvcore/MathTypes.cpp
  cvcore/src/core/cvcore/Tensor.cpp
  cvcore/src/core/utility/CVError.cpp
  cvcore/src/core/utility/Instrumentation.cpp
  cvcore/src/core/utility/Memory.cpp
  cvcore/src/core/utility/ProfileUtils.cpp

  # Inferencer (ESS only)
  cvcore/src/inferencer/tensorrt/TensorRTInferencer.cpp
  cvcore/src/inferencer/Inferencer.cpp
  cvcore/src/inferencer/Errors.cpp
  cvcore/src/inferencer/tensorrt/TensorRTUtils.h
  cvcore/src/inferencer/tensorrt/TensorRTUtils.cpp
  cvcore/src/inferencer/tensorrt/TensorRTInferencer.h

  # TRTBackend
  cvcore/src/trtbackend/TRTBackend.cpp
)

target_include_directories(cvcore_ess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/include)
target_compile_options(cvcore_ess PUBLIC -fPIC)
target_link_libraries(cvcore_ess PUBLIC
  CUDA::cudart
  CUDA::nppc
  CUDA::nppial
  CUDA::nppicc
  CUDA::nppidei
  CUDA::nppif
  CUDA::nppig
  CUDA::nppisu
  TENSORRT::nvinfer
)

add_library(ess_3dv STATIC
  3dv/src/ESS.cpp
  3dv/src/ESSPostProcess.cpp
  3dv/src/ESSPreProcess.cpp
)
target_include_directories(ess_3dv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3dv/include)
target_compile_options(ess_3dv PUBLIC -fPIC)
target_link_libraries(ess_3dv PUBLIC
  cvcore_ess
)
target_link_libraries(gxf_cvcore_ess
  PUBLIC
    GXF::cuda
    GXF::multimedia
    yaml-cpp
  PRIVATE
    cvcore_ess
    ess_3dv
)
