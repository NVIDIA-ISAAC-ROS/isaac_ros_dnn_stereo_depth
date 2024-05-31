// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/tensorops/core/CVError.h"

namespace cvcore {
namespace inferencer {

/*
 * Enum class describing the inference error codes.
 */
enum class InferencerErrorCode : std::int32_t {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    INVALID_OPERATION,
    NOT_IMPLEMENTED,
    TRITON_SERVER_NOT_READY,
    TRITON_CUDA_SHARED_MEMORY_ERROR,
    TRITON_INFERENCE_ERROR,
    TRITON_REGISTER_LAYER_ERROR,
    TENSORRT_INFERENCE_ERROR,
    TENSORRT_ENGINE_ERROR
};

}  // namespace inferencer
}  // namespace cvcore

namespace std {

template<>
struct is_error_code_enum<cvcore::inferencer::InferencerErrorCode> : true_type {
};

}  // namespace std

namespace cvcore {
namespace inferencer {

std::error_code make_error_code(InferencerErrorCode) noexcept;

}  // namespace inferencer
}  // namespace cvcore
