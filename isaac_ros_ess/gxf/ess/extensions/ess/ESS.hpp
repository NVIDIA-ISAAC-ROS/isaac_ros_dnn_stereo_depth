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

#ifndef NVIDIA_CVCORE_ESS_HPP
#define NVIDIA_CVCORE_ESS_HPP

#include <cstring>
#include <fstream>
#include <vector>

#include <cv/ess/ESS.h>

#include "extensions/tensor_ops/ImageAdapter.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace cvcore {

// CV-Core ESS GXF Codelet
class ESS : public gxf::Codelet {
public:
  ESS()  = default;
  ~ESS() = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override {
    return GXF_SUCCESS;
  }
  gxf_result_t deinitialize() override {
    return GXF_SUCCESS;
  }

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

private:
  // cvcore image pre-processing params for ESS
  ::cvcore::ImagePreProcessingParams preProcessorParams;
  // cvcore model input params for ESS
  ::cvcore::ModelInputParams modelInputParams;
  // cvcore inference params for ESS
  ::cvcore::ModelInferenceParams inferenceParams;
  // extra params for ESS
  ::cvcore::ess::ESSPreProcessorParams extraParams;
  // cvcore ESS object
  std::unique_ptr<::cvcore::ess::ESS> objESS;

  // The name of the input left image tensor
  gxf::Parameter<std::string> left_image_name_;
  // The name of the input right image tensor
  gxf::Parameter<std::string> right_image_name_;
  // The name of the output tensor
  gxf::Parameter<std::string> output_name_;
  // The Cuda Stream pool for allocate cuda stream
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> stream_pool_;
  // Data allocator to create a tensor
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  // Data receiver to get left image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> left_image_receiver_;
  // Data receiver to get right image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> right_image_receiver_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  // Image adapter for output image
  gxf::Parameter<gxf::Handle<tensor_ops::ImageAdapter>> output_adapter_;

  // Pre-processing params for ESS
  gxf::Parameter<std::string> image_type_;
  gxf::Parameter<std::vector<float>> pixel_mean_;
  gxf::Parameter<std::vector<float>> normalization_;
  gxf::Parameter<std::vector<float>> standard_deviation_;

  // Model input params for ESS
  gxf::Parameter<int> max_batch_size_;
  gxf::Parameter<int> input_layer_width_;
  gxf::Parameter<int> input_layer_height_;
  gxf::Parameter<std::string> model_input_type_;

  // Inference params for ESS
  gxf::Parameter<std::string> engine_file_path_;
  gxf::Parameter<std::vector<std::string>> input_layers_name_;
  gxf::Parameter<std::vector<std::string>> output_layers_name_;

  // Extra Pre-process param
  gxf::Parameter<std::string> preprocess_type_;

  // Output params
  gxf::Parameter<size_t> output_width_;
  gxf::Parameter<size_t> output_height_;

  // Decide which timestamp to pass down
  gxf::Parameter<int> timestamp_policy_;

  gxf::Handle<gxf::CudaStream> cuda_stream_ = nullptr;
};

} // namespace cvcore
} // namespace nvidia

#endif
