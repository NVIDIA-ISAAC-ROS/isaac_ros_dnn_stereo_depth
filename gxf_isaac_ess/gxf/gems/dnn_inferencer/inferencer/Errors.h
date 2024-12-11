/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

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
