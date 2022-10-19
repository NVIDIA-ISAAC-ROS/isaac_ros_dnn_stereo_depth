# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute path to the ESS engine plan.'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_disparity::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path}],
        remappings=[('left/image_rect', 'rgb_left'),
                    ('left/camera_info', 'camera_info_left'),
                    ('right/image_rect', 'rgb_right'),
                    ('right/camera_info', 'camera_info_right')])

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
            'use_color': True,
            'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'rgb_left'),
                    ('left/camera_info', 'camera_info_left'),
                    ('right/camera_info', 'camera_info_right')])

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[disparity_node, pointcloud_node],
        output='screen'
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_ess'), 'config', 'isaac_ros_ess_isaac_sim.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')

    return (launch.LaunchDescription(launch_args + [container, rviz_node]))
