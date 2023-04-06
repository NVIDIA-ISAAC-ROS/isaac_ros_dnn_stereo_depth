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

#include "cv/inferencer/Errors.h"

#ifndef __cpp_lib_to_underlying
namespace std {
template<typename Enum>
constexpr underlying_type_t<Enum> to_underlying(Enum e) noexcept
{
    return static_cast<underlying_type_t<Enum>>(e);
}
};     // namespace std
#endif // __cpp_lib_to_underlying

namespace cvcore { namespace inferencer {

namespace detail {
struct InferencerErrorCategory : std::error_category
{
    virtual const char *name() const noexcept override final
    {
        return "cvcore-inferencer-error";
    }

    virtual std::string message(int value) const override final
    {
        std::string result;

        switch (value)
        {
        case std::to_underlying(InferencerErrorCode::SUCCESS):
            result = "(SUCCESS) No errors detected";
            break;
        case std::to_underlying(InferencerErrorCode::INVALID_ARGUMENT):
            result = "(INVALID_ARGUMENT) Invalid config parameter or input argument";
            break;
        case std::to_underlying(InferencerErrorCode::INVALID_OPERATION):
            result = "(INVALID_OPERATION) Invalid operation performed";
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_SERVER_NOT_READY):
            result = "(TRITON_SERVER_NOT_READY) Triton server is not live or the serverUrl is incorrect";
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR):
            result = "(TRITON_CUDA_SHARED_MEMORY_ERROR) Unable to map/unmap cuda shared memory for triton server";
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_INFERENCE_ERROR):
            result = "(TRITON_INFERENCE_ERROR) Error during inference using triton API";
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_REGISTER_LAYER_ERROR):
            result = "(TRITON_REGISTER_LAYER_ERROR) Error when setting input or output layers";
            break;
        case std::to_underlying(InferencerErrorCode::TENSORRT_INFERENCE_ERROR):
            result = "(TENSORRT_INFERENCE_ERROR) Error when running TensorRT enqueue/execute";
            break;
        default:
            result = "(Unrecognized Condition) Value " + std::to_string(value) +
                     " does not map to known error code literal " +
                     " defined by cvcore::inferencer::InferencerErrorCode";
            break;
        }

        return result;
    }

    virtual std::error_condition default_error_condition(int code) const noexcept override final
    {
        std::error_condition result;

        switch (code)
        {
        case std::to_underlying(InferencerErrorCode::SUCCESS):
            result = ErrorCode::SUCCESS;
            break;
        case std::to_underlying(InferencerErrorCode::INVALID_ARGUMENT):
            result = ErrorCode::INVALID_ARGUMENT;
            break;
        case std::to_underlying(InferencerErrorCode::INVALID_OPERATION):
            result = ErrorCode::INVALID_OPERATION;
            break;
        case std::to_underlying(InferencerErrorCode::NOT_IMPLEMENTED):
            result = ErrorCode::NOT_IMPLEMENTED;
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_SERVER_NOT_READY):
            result = ErrorCode::NOT_READY;
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_CUDA_SHARED_MEMORY_ERROR):
            result = ErrorCode::DEVICE_ERROR;
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_INFERENCE_ERROR):
            result = ErrorCode::INVALID_OPERATION;
            break;
        case std::to_underlying(InferencerErrorCode::TENSORRT_INFERENCE_ERROR):
            result = ErrorCode::INVALID_OPERATION;
            break;
        case std::to_underlying(InferencerErrorCode::TRITON_REGISTER_LAYER_ERROR):
            result = ErrorCode::INVALID_OPERATION;
            break;
        default:
            result = ErrorCode::NOT_IMPLEMENTED;
            break;
        }

        return result;
    }
};
} // namespace detail

const detail::InferencerErrorCategory errorCategory{};

std::error_code make_error_code(InferencerErrorCode ec) noexcept
{
    return {std::to_underlying(ec), errorCategory};
}
}} // namespace cvcore::inferencer
