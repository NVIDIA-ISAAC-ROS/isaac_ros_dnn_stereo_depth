# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import subprocess


class ESSEngineGenerator:

    def __init__(self,
                 etlt_model,
                 arch=''):
        self.etlt_model = etlt_model
        if not arch:
            self.arch = platform.machine()
            print('Architecture of the target platform is {}'.format(self.arch))
        else:
            self.arch = arch

    def generate(self):
        supported_arch = ['x86_64', 'aarch64']
        model_file = os.path.abspath(self.etlt_model)
        if self.arch not in supported_arch:
            print('Unsupported architecture: {}. Supported architectures are:'
                  '{}'.format(self.arch, supported_arch))
            return
        elif os.path.exists(os.path.abspath(self.etlt_model)):
            plugin = (os.path.dirname(model_file) + '/plugins/' +
                      self.arch + '/ess_plugins.so')
            engine_file = model_file.replace('.etlt', '.engine')

            response = subprocess.call('LD_PRELOAD=' + plugin +
                                       ' tao-converter -k ess -t fp16' +
                                       ' -e ' + engine_file +
                                       ' -o output_left,output_conf ' +
                                       self.etlt_model, shell=True)

            if response == 0:
                print('Engine file for ESS model{} is generated!'
                      .format(self.etlt_model))
            else:
                print('Failed to generate engine file for model {}'
                      .format(self.etlt_model))
        else:
            print('ESS etlt model is not found.')
