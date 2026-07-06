# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Core launch fragment for the DNNStereoDecoder component.

Provides a fragment that creates the `ComposableNode` for the decoder and its launch arguments.
This can be included or composed into larger application pipelines.
"""

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


class IsaacROSDnnStereoDecoderLaunchFragment(IsaacROSLaunchFragment):
    """Launch fragment for the DNNStereoDecoder `ComposableNode`."""

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        """Return a mapping of node names to `ComposableNode` descriptions for the decoder."""
        # Decoder parameters
        min_disparity = LaunchConfiguration('min_disparity')
        max_disparity = LaunchConfiguration('max_disparity')

        return {
            # Disparity decoder node
            'dnn_stereo_decoder_node': ComposableNode(
                name='dnn_stereo_decoder',
                package='isaac_ros_dnn_stereo_decoder',
                plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
                parameters=[{
                    'disparity_tensor_name': 'disparity',
                    'min_disparity': min_disparity,
                    'max_disparity': max_disparity,
                }],
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        """Return launch actions (declare arguments) for the decoder."""
        return {
            'min_disparity': DeclareLaunchArgument(
                'min_disparity',
                default_value='0.0',
                description='Minimum disparity value (inclusive)'
            ),
            'max_disparity': DeclareLaunchArgument(
                'max_disparity',
                default_value='10000.0',
                description='Maximum disparity value (inclusive)'
            )
        }


def generate_launch_description() -> launch.LaunchDescription:
    """Create and return a `LaunchDescription` that starts the decoder container."""
    dnn_stereo_decoder_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='dnn_stereo_decoder_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSDnnStereoDecoderLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription(
        [dnn_stereo_decoder_container] +
        IsaacROSDnnStereoDecoderLaunchFragment.get_launch_actions().values())
