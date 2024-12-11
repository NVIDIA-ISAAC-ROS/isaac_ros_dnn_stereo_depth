# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute path to the ESS engine plan.'),
        DeclareLaunchArgument(
            'threshold',
            default_value='0.4',
            description='Threshold value ranges between 0.0 and 1.0 '
            'for filtering disparity with confidence.'),
        DeclareLaunchArgument(
            'input_layer_width',
            default_value='960',
            description='Input layer width'),
        DeclareLaunchArgument(
            'input_layer_height',
            default_value='576',
            description='Input layer height'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')
    threshold = LaunchConfiguration('threshold')
    input_layer_width = LaunchConfiguration('input_layer_width')
    input_layer_height = LaunchConfiguration('input_layer_height')

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold,
                     'input_layer_width': input_layer_width,
                     'input_layer_height': input_layer_height}],
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[disparity_node],
        output='screen'
    )

    return (launch.LaunchDescription(launch_args + [container]))
