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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


class IsaacROSEssLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        engine_file_path = LaunchConfiguration('engine_file_path')
        threshold = LaunchConfiguration('threshold')
        input_layer_width = LaunchConfiguration('input_layer_width')
        input_layer_height = LaunchConfiguration('input_layer_height')

        return {
            'left_resize_node': ComposableNode(
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                name='left_resize',
                parameters=[{
                        'output_width': 960,
                        'output_height': 576,
                        'keep_aspect_ratio': False
                }],
                remappings=[
                    ('camera_info', 'left/camera_info_rect'),
                    ('image', 'left/image_rect'),
                    ('resize/camera_info', 'left/camera_info_resize'),
                    ('resize/image', 'left/image_resize')]
            ),
            'right_resize_node': ComposableNode(
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                name='right_resize',
                parameters=[{
                        'output_width': 960,
                        'output_height': 576,
                        'keep_aspect_ratio': False
                }],
                remappings=[
                    ('camera_info', 'right/camera_info_rect'),
                    ('image', 'right/image_rect'),
                    ('resize/camera_info', 'right/camera_info_resize'),
                    ('resize/image', 'right/image_resize')]
            ),
            'ess_disparity': ComposableNode(
                package='isaac_ros_ess',
                plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
                name='ess_disparity',
                namespace='',
                parameters=[{
                    'engine_file_path': engine_file_path,
                    'threshold': threshold,
                    'input_layer_width': input_layer_width,
                    'input_layer_height': input_layer_height,
                }],
                remappings=[
                    ('left/image_rect', 'left/image_resize'),
                    ('right/image_rect', 'right/image_resize'),
                    ('left/camera_info', 'left/camera_info_resize'),
                    ('right/camera_info', 'right/camera_info_resize'),
                ],
            ),
            'disparity_to_depth_node': ComposableNode(
                name='DisparityToDepthNode',
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        return {
            'engine_file_path': DeclareLaunchArgument(
                'engine_file_path',
                default_value=''
            ),
            'threshold': DeclareLaunchArgument(
                'threshold',
                default_value='0.35'
            ),
            'input_layer_width': DeclareLaunchArgument(
                'input_layer_width',
                default_value='960'
            ),
            'input_layer_height': DeclareLaunchArgument(
                'input_layer_height',
                default_value='576'
            ),
        }


def generate_launch_description():
    """Launch file to bring up ESS node standalone."""
    ess_launch_container = ComposableNodeContainer(
        name='ess_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSEssLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription([ess_launch_container])
