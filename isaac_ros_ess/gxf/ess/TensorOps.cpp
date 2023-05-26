// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/ess/ESS.hpp"
#include "extensions/tensor_ops/CameraModel.hpp"
#include "extensions/tensor_ops/ConvertColorFormat.hpp"
#include "extensions/tensor_ops/CropAndResize.hpp"
#include "extensions/tensor_ops/Frame3D.hpp"
#include "extensions/tensor_ops/ImageAdapter.hpp"
#include "extensions/tensor_ops/InterleavedToPlanar.hpp"
#include "extensions/tensor_ops/Normalize.hpp"
#include "extensions/tensor_ops/Reshape.hpp"
#include "extensions/tensor_ops/Resize.hpp"
#include "extensions/tensor_ops/TensorOperator.hpp"
#include "extensions/tensor_ops/TensorStream.hpp"
#include "extensions/tensor_ops/Undistort.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(
  0x6eae64ff97a94d9b, 0xb324f85e6a98a75a, "NvCvTensorOpsExtension",
  "Generic CVCORE tensor_ops interfaces", "Nvidia_Gxf", "3.1.0", "LICENSE");

GXF_EXT_FACTORY_ADD(
  0xd073a92344ba4b82, 0xbd0f18f4996048e8, nvidia::cvcore::tensor_ops::CameraModel,
  nvidia::gxf::Component,
  "Construct Camera distortion model / Camera intrinsic compatible with CVCORE");

GXF_EXT_FACTORY_ADD(
  0x6c9419223e4b4c2c, 0x899a4d65279c6508, nvidia::cvcore::tensor_ops::Frame3D,
  nvidia::gxf::Component,
  "Construct Camera extrinsic compatible with CVCORE");

GXF_EXT_FACTORY_ADD(
  0xd94385e5b35b4635, 0x9adb0d214a3865f7, nvidia::cvcore::tensor_ops::TensorStream,
  nvidia::gxf::Component, "Wrapper of CVCORE ITensorOperatorStream/ITensorOperatorContext");

GXF_EXT_FACTORY_ADD(
  0xd0c4ddad486a4a92, 0xb69c8a5304b205ea, nvidia::cvcore::tensor_ops::ImageAdapter,
  nvidia::gxf::Component, "Utility component for conversion between message and cvcore image type");

GXF_EXT_FACTORY_ADD(
  0xadebc792bd0b4a57, 0x99c1405fd2ea0728, nvidia::cvcore::tensor_ops::StreamUndistort,
  nvidia::gxf::Codelet, "Codelet for stream image undistortion in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0xa58141ac7eca4ea6, 0x9b545446fe379a12, nvidia::cvcore::tensor_ops::Resize, nvidia::gxf::Codelet,
  "Codelet for image resizing in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0xeb8b5f5b36d44b49, 0x81f959fd28e6f678, nvidia::cvcore::tensor_ops::StreamResize,
  nvidia::gxf::Codelet, "Codelet for stream image resizing in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0x4a7ff422de3841bd, 0x9e743ac10d9294b7, nvidia::cvcore::tensor_ops::CropAndResize,
  nvidia::gxf::Codelet, "Codelet for crop and resizing operation in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0x7018f0b9034c462c, 0xa9fbaf7ee012974a, nvidia::cvcore::tensor_ops::Normalize,
  nvidia::gxf::Codelet,
  "Codelet for image normalization in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0x269d4237f3c3479e, 0xbcca9ecc44c71a71, nvidia::cvcore::tensor_ops::InterleavedToPlanar,
  nvidia::gxf::Codelet, "Codelet for convert interleaved image to planar image in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0xfc4d7b4d8fcc4dab, 0xa286056e0fcafa79, nvidia::cvcore::tensor_ops::ConvertColorFormat,
  nvidia::gxf::Codelet, "Codelet for image color conversion in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0x5ab4a4d8f7a34553, 0xa90be52660b076fe, nvidia::cvcore::tensor_ops::StreamConvertColorFormat,
  nvidia::gxf::Codelet, "Codelet for stream image color conversion in tensor_ops");

GXF_EXT_FACTORY_ADD(
  0x26789b7d5a8d4e85, 0x86b845ec5f4cd12b, nvidia::cvcore::tensor_ops::Reshape, nvidia::gxf::Codelet,
  "Codelet for image reshape in tensor_ops");
GXF_EXT_FACTORY_ADD(
  0x1aa1eea914344aff, 0x97fddaaedb594121, nvidia::cvcore::ESS, nvidia::gxf::Codelet,
  "ESS GXF Extension");
GXF_EXT_FACTORY_END()
