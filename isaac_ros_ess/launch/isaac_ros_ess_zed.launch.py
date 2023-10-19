# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from launch.substitutions import Command, LaunchConfiguration
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
            default_value='0.9',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')
    threshold = LaunchConfiguration('threshold')

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold}],
        remappings=[
            ('left/camera_info', 'zed_node/left/camera_info_resize'),
            ('left/image_rect', 'zed_node/left/image_rect_color_rgb_resize'),
            ('right/camera_info', 'zed_node/right/camera_info_resize'),
            ('right/image_rect', 'zed_node/right/image_rect_color_rgb_resize')
        ]
    )

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': True,
                'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'zed_node/left/image_rect_color_rgb_resize'),
                    ('left/camera_info', 'zed_node/left/camera_info_resize'),
                    ('right/camera_info', 'zed_node/right/camera_info_resize')]
    )

    image_format_converter_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_left',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'zed_node/left/image_rect_color'),
            ('image', 'zed_node/left/image_rect_color_rgb')]
    )

    image_format_converter_node_right = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_right',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'zed_node/right/image_rect_color'),
            ('image', 'zed_node/right/image_rect_color_rgb')]
    )

    image_resize_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize_left',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'zed_node/left/camera_info'),
            ('image', 'zed_node/left/image_rect_color_rgb'),
            ('resize/camera_info', 'zed_node/left/camera_info_resize'),
            ('resize/image', 'zed_node/left/image_rect_color_rgb_resize')]
    )

    image_resize_node_right = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize_right',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'zed_node/right/camera_info'),
            ('image', 'zed_node/right/image_rect_color_rgb'),
            ('resize/camera_info', 'zed_node/right/camera_info_resize'),
            ('resize/image', 'zed_node/right/image_rect_color_rgb_resize')]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[image_format_converter_node_left,
                                      image_format_converter_node_right,
                                      image_resize_node_left,
                                      image_resize_node_right,
                                      disparity_node,
                                      pointcloud_node],
        output='screen'
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_ess'), 'config', 'isaac_ros_ess_zed.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')

    # Define LaunchDescription variable
    ld = launch.LaunchDescription(launch_args + [container, rviz])

    # https://www.stereolabs.com/docs/ros2/ros2-composition/
    # Camera model (force value)
    camera_model = 'zed2i'

    # URDF/xacro file to be loaded by the Robot State Publisher node
    xacro_path = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'urdf', 'zed_descr.urdf.xacro'
    )

    # ZED Configurations to be loaded by ZED Node
    config_common = os.path.join(
        get_package_share_directory('isaac_ros_ess'),
        'config',
        'zed.yaml'
    )

    config_camera = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'config',
        camera_model + '.yaml'
    )

    # Robot State Publisher node
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='zed_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(
                [
                    'xacro', ' ', xacro_path, ' ',
                    'camera_name:=', camera_model, ' ',
                    'camera_model:=', camera_model
                ])
        }]
    )

    # ZED node using manual composition
    zed_node = Node(
        package='zed_wrapper',
        executable='zed_wrapper',
        output='screen',
        parameters=[
            config_common,  # Common parameters
            config_camera,  # Camera related parameters
        ]
    )

    # Add nodes to LaunchDescription
    ld.add_action(rsp_node)
    ld.add_action(zed_node)

    return ld
