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

#pragma once
#include "engine/core/math/types.hpp"
#include "cuda_runtime.h"

namespace nvidia {
namespace isaac {

void threshold_image_cuda(const float* image, const float* mask,
                          float* output, const int2 image_size, const float threshold,
                          const float fill_value);

void threshold_image_cuda(const unsigned char * image, const float * mask,
                          unsigned char * output, const int2 image_size, const float threshold,
                          const uchar3 fill_value);
}  // namespace isaac
}  // namespace nvidia
