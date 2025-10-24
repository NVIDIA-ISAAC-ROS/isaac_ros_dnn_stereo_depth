// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>

#include "common/fixed_string.hpp"
#include "common/span.hpp"
#include "gxf/core/expected.hpp"

namespace nvidia {
namespace isaac {

/// The SHA256 class Generates the SHA256 sum of a given string
/// Adapted from https://www.programmingalgorithms.com/algorithm/sha256
class SHA256 {
 public:
  static constexpr size_t HASH_LENGTH = 32;
  using Result = std::array<byte, HASH_LENGTH>;
  using String = FixedString<HASH_LENGTH * 2>;

  static gxf::Expected<Result> Hash(const Span<const byte> data);
  static gxf::Expected<String> Hash(const Span<const char> data);

  SHA256() : context_{}, hash_{} { Initialize(context_); }

  /// Reset hasher to be fed again
  void reset() { Initialize(context_); }

  /// Hash a given array
  gxf::Expected<void> hashData(const Span<const byte> data) { return Update(context_, data); }

  /// Finalize computation of the hash, i.e. make solution available through `hash()`
  gxf::Expected<void> finalize() {
    return Finalize(context_)
        .map([&](Result result) {
          hash_ = result;
          return gxf::Success;
        });
  }

  /// Return hashed result
  const Result& hash() const { return hash_; }
  /// Return base64 encoding of the hash
  gxf::Expected<String> toString() const { return ToString(hash_); }

 private:
  struct SHA256_CTX {
    uint8_t data[64];
    uint32_t datalen;
    uint32_t bitlen[2];
    uint32_t state[8];
  };

  static void Initialize(SHA256_CTX& ctx);
  static gxf::Expected<void> Update(SHA256_CTX& ctx, const Span<const byte> data);
  static gxf::Expected<void> Transform(SHA256_CTX& ctx, const Span<const byte> data);
  static gxf::Expected<Result> Finalize(SHA256_CTX& ctx);
  static gxf::Expected<String> ToString(const Result& hash);

  SHA256_CTX context_;
  Result hash_;
};

}  // namespace isaac
}  // namespace nvidia
