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

#include "extensions/ess/components/ess_inference.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xce7c6985267a4ec7, 0xa073030e16e49f29, "ESS",
                         "Extension containing ESS related components",
                         "Isaac SDK", "2.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x1aa1eea914344afe, 0x97fddaaedb594120,
                    nvidia::isaac::ESSInference, nvidia::gxf::Codelet,
                    "ESS GXF Extension");

GXF_EXT_FACTORY_END()
