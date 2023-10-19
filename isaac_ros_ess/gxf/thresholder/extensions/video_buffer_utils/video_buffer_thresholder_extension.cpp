// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/video_buffer_utils/components/video_buffer_thresholder.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

// This is an extension made out of isaac::video_buffer_utils
GXF_EXT_FACTORY_SET_INFO(
  0xaf5b39a0485f4f28, 0x83eb359d36049b68, "ThresholderExtension",
  "Extension containing a video buffer threaholder extension", "NVIDIA", "0.0.1",
  "LICENSE");

GXF_EXT_FACTORY_ADD(
  0xc757549c430d11ee, 0xbb0073c7d954a493,
  nvidia::isaac::VideoBufferThresholder, nvidia::gxf::Codelet,
  "Thresholds an input VideoBuffers given a mask and a threshold for the mask");

GXF_EXT_FACTORY_END()
