# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        DeclareLaunchArgument(
            'threshold',
            default_value='0.0',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')
    threshold = LaunchConfiguration('threshold')

    image_resize_node_right = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize_node_right',
        parameters=[{
                'output_width': 960,
                'output_height': 576,
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('camera_info', 'front_stereo_camera/right_rgb/camerainfo'),
            ('image', 'front_stereo_camera/right_rgb/image_raw'),
            ('resize/camera_info', 'front_stereo_camera/right_rgb/camerainfo_resize'),
            ('resize/image', 'front_stereo_camera/right_rgb/image_resize')]
    )

    image_resize_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize_node_left',
        parameters=[{
                'output_width': 960,
                'output_height': 576,
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('camera_info', 'front_stereo_camera/left_rgb/camerainfo'),
            ('image', 'front_stereo_camera/left_rgb/image_raw'),
            ('resize/camera_info', 'front_stereo_camera/left_rgb/camerainfo_resize'),
            ('resize/image', 'front_stereo_camera/left_rgb/image_resize')]
    )

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold}],
        remappings=[('left/image_rect', 'front_stereo_camera/left_rgb/image_resize'),
                    ('left/camera_info', 'front_stereo_camera/left_rgb/camerainfo_resize'),
                    ('right/image_rect', 'front_stereo_camera/right_rgb/image_resize'),
                    ('right/camera_info', 'front_stereo_camera/right_rgb/camerainfo_resize')])

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
            'use_color': True,
            'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'front_stereo_camera/left_rgb/image_resize'),
                    ('left/camera_info', 'front_stereo_camera/left_rgb/camerainfo_resize'),
                    ('right/camera_info', 'front_stereo_camera/right_rgb/camerainfo_resize')])

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[disparity_node, pointcloud_node,
                                      image_resize_node_left, image_resize_node_right
                                      ],
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
