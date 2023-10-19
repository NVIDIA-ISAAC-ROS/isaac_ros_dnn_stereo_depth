// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gems/hash/sha256.hpp"

#include <cstring>

namespace nvidia {
namespace isaac {

namespace {

constexpr uint32_t K[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
                             0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                             0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
                             0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                             0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                             0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
                             0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
                             0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                             0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

#define DBL_INT_ADD(a, b, c)    \
  if ((a) > 0xFFFFFFFF - (c)) { \
    ++(b);                      \
  }                             \
  (a) += (c);

#define ROTLEFT(a, b)  (((a) << (b)) | ((a) >> (32 - (b))))
#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))

#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define EP0(x)  (ROTRIGHT((x),  2) ^ ROTRIGHT((x), 13) ^ ROTRIGHT((x), 22))
#define EP1(x)  (ROTRIGHT((x),  6) ^ ROTRIGHT((x), 11) ^ ROTRIGHT((x), 25))
#define SIG0(x) (ROTRIGHT((x),  7) ^ ROTRIGHT((x), 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT((x), 17) ^ ROTRIGHT((x), 19) ^ ((x) >> 10))

}  // namespace

gxf::Expected<SHA256::Result> SHA256::Hash(const Span<const byte> data) {
  SHA256 hasher;
  return hasher.hashData(data)
      .and_then([&]() { return hasher.finalize(); })
      .substitute(hasher.hash());
}

gxf::Expected<SHA256::String> SHA256::Hash(const Span<const char> data) {
  auto span = Span<const byte>(reinterpret_cast<const byte*>(data.data()), data.size());
  return Hash(span).map(ToString);
}

void SHA256::Initialize(SHA256_CTX& ctx) {
  ctx.datalen   = 0;
  ctx.bitlen[0] = 0;
  ctx.bitlen[1] = 0;
  ctx.state[0]  = 0x6a09e667;
  ctx.state[1]  = 0xbb67ae85;
  ctx.state[2]  = 0x3c6ef372;
  ctx.state[3]  = 0xa54ff53a;
  ctx.state[4]  = 0x510e527f;
  ctx.state[5]  = 0x9b05688c;
  ctx.state[6]  = 0x1f83d9ab;
  ctx.state[7]  = 0x5be0cd19;
}

gxf::Expected<void> SHA256::Update(SHA256_CTX& ctx, const Span<const byte> data) {
  for (auto element : data) {
    if (!element) {
      return gxf::Unexpected{GXF_FAILURE};
    }
    ctx.data[ctx.datalen] = element.value();
    ctx.datalen++;
    if (ctx.datalen == 64) {
      auto result = Transform(ctx, Span<const byte>(ctx.data));
      if (!result) {
        return gxf::ForwardError(result);
      }
      DBL_INT_ADD(ctx.bitlen[0], ctx.bitlen[1], 512);
      ctx.datalen = 0;
    }
  }
  return gxf::Success;
}

gxf::Expected<void> SHA256::Transform(SHA256_CTX& ctx, const Span<const byte> data) {
  uint32_t m[64];
  uint32_t i = 0;
  uint32_t j = 0;
  for (; i < 16; i++) {
    if (!data[j] || !data[j + 1] || !data[j + 2] || !data[j + 3]) {
      return gxf::Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
    m[i] = (static_cast<uint32_t>(data[j].value())     << 24 & 0xFF000000)
         | (static_cast<uint32_t>(data[j + 1].value()) << 16 & 0x00FF0000)
         | (static_cast<uint32_t>(data[j + 2].value()) <<  8 & 0x0000FF00)
         | (static_cast<uint32_t>(data[j + 3].value()) <<  0 & 0x000000FF);
    j += 4;
  }
  for (; i < 64; i++) {
    m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
  }

  uint32_t a = ctx.state[0];
  uint32_t b = ctx.state[1];
  uint32_t c = ctx.state[2];
  uint32_t d = ctx.state[3];
  uint32_t e = ctx.state[4];
  uint32_t f = ctx.state[5];
  uint32_t g = ctx.state[6];
  uint32_t h = ctx.state[7];
  uint32_t t1;
  uint32_t t2;
  for (i = 0; i < 64; ++i) {
    t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
    t2 = EP0(a) + MAJ(a, b, c);
    h  = g;
    g  = f;
    f  = e;
    e  = d + t1;
    d  = c;
    c  = b;
    b  = a;
    a  = t1 + t2;
  }

  ctx.state[0] += a;
  ctx.state[1] += b;
  ctx.state[2] += c;
  ctx.state[3] += d;
  ctx.state[4] += e;
  ctx.state[5] += f;
  ctx.state[6] += g;
  ctx.state[7] += h;

  return gxf::Success;
}

gxf::Expected<SHA256::Result> SHA256::Finalize(SHA256_CTX& ctx) {
  uint32_t i = ctx.datalen;

  if (ctx.datalen < 56) {
    ctx.data[i] = 0x80;
    i++;

    while (i < 56) {
      ctx.data[i] = 0x00;
      i++;
    }
  } else {
    ctx.data[i] = 0x80;
    i++;

    while (i < 64) {
      ctx.data[i] = 0x00;
      i++;
    }

    auto result = Transform(ctx, Span<const byte>(ctx.data));
    if (!result) {
      return gxf::ForwardError(result);
    }
    std::memset(ctx.data, 0, 56);
  }

  DBL_INT_ADD(ctx.bitlen[0], ctx.bitlen[1], ctx.datalen * 8);
  ctx.data[63] = static_cast<uint8_t>(ctx.bitlen[0] >>  0 & 0x000000FF);
  ctx.data[62] = static_cast<uint8_t>(ctx.bitlen[0] >>  8 & 0x000000FF);
  ctx.data[61] = static_cast<uint8_t>(ctx.bitlen[0] >> 16 & 0x000000FF);
  ctx.data[60] = static_cast<uint8_t>(ctx.bitlen[0] >> 24 & 0x000000FF);
  ctx.data[59] = static_cast<uint8_t>(ctx.bitlen[1] >>  0 & 0x000000FF);
  ctx.data[58] = static_cast<uint8_t>(ctx.bitlen[1] >>  8 & 0x000000FF);
  ctx.data[57] = static_cast<uint8_t>(ctx.bitlen[1] >> 16 & 0x000000FF);
  ctx.data[56] = static_cast<uint8_t>(ctx.bitlen[1] >> 24 & 0x000000FF);

  auto result = Transform(ctx, Span<const byte>(ctx.data));
  if (!result) {
    return gxf::ForwardError(result);
  }

  Result hash;
  for (i = 0; i < 4; ++i) {
    hash[i]      = static_cast<uint8_t>(ctx.state[0] >> (24 - i * 8) & 0x000000FF);
    hash[i + 4]  = static_cast<uint8_t>(ctx.state[1] >> (24 - i * 8) & 0x000000FF);
    hash[i + 8]  = static_cast<uint8_t>(ctx.state[2] >> (24 - i * 8) & 0x000000FF);
    hash[i + 12] = static_cast<uint8_t>(ctx.state[3] >> (24 - i * 8) & 0x000000FF);
    hash[i + 16] = static_cast<uint8_t>(ctx.state[4] >> (24 - i * 8) & 0x000000FF);
    hash[i + 20] = static_cast<uint8_t>(ctx.state[5] >> (24 - i * 8) & 0x000000FF);
    hash[i + 24] = static_cast<uint8_t>(ctx.state[6] >> (24 - i * 8) & 0x000000FF);
    hash[i + 28] = static_cast<uint8_t>(ctx.state[7] >> (24 - i * 8) & 0x000000FF);
  }

  return hash;
}

gxf::Expected<SHA256::String> SHA256::ToString(const Result& hash) {
  constexpr char INT2HEX[] = { '0', '1', '2', '3', '4', '5', '6', '7',
                               '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

  String text;
  for (auto&& value : hash) {
    const uint8_t upper = static_cast<uint8_t>(value >> 4 & 0x0F);
    const uint8_t lower = static_cast<uint8_t>(value >> 0 & 0x0F);

    auto result = text.append(INT2HEX[upper])
        .and_then([&]() { return text.append(INT2HEX[lower]); });
    if (!result) {
      return gxf::Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE};
    }
  }

  return text;
}

}  // namespace isaac
}  // namespace nvidia
