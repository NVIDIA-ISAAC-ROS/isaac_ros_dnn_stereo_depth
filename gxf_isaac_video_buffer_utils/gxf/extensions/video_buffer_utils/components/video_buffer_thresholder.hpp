// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <string>
#include <vector>

#include "extensions/gxf_helpers/parameter_parser_isaac.hpp"
#include "extensions/gxf_helpers/parameter_wrapper_isaac.hpp"
#include "gems/core/math/types.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace isaac {

// Image threshold for VideoBuffers
//
// This codelet consumes a VideoBuffer image (float or RGB), a VideoBuffer mask (float), a threshold
// (float) for the mask, and a fill value (float or RGB) to "paint" the pixels in the image that
// are under or equal to the threshold in the mask image:
// if mask[index] <= threshold: image[index] = fill_value
class VideoBufferThresholder : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t stop() override;
  gxf_result_t tick() override;

 private:
  gxf_result_t thresholdImage(
    const gxf::Handle<gxf::VideoBuffer> image_video_buffer,
    const gxf::Handle<gxf::VideoBuffer> mask_video_buffer,
    gxf::Handle<gxf::VideoBuffer> output_video_buffer,
    cudaStream_t cuda_stream) const;
  gxf_result_t validateImageMessage(const gxf::Handle<gxf::VideoBuffer> image) const;
  gxf_result_t validateMaskMessage(const gxf::Handle<gxf::VideoBuffer> image,
                                   const gxf::Handle<gxf::VideoBuffer> mask) const;
  uchar3 toUchar3(const Vector3i& vector3i) const;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> image_input_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> mask_input_;
  gxf::Parameter<float> threshold_;
  gxf::Parameter<float> fill_value_float_;
  gxf::Parameter<Vector3i> fill_value_rgb_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> masked_output_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<std::string> video_buffer_name_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> stream_pool_;

  gxf::Handle<gxf::CudaStream> cuda_stream_ = nullptr;
};

}  // namespace isaac
}  // namespace nvidia
