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
            default_value='0.35',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),
        DeclareLaunchArgument(
            'module_id',
            default_value='2',
            description='Index specifying the stereo camera module to use.'),
        DeclareLaunchArgument(
            'output_width',
            default_value='960',
            description='ESS model output width.'),
        DeclareLaunchArgument(
            'output_height',
            default_value='576',
            description='ESS model output height.'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')
    threshold = LaunchConfiguration('threshold')
    module_id = LaunchConfiguration('module_id')
    output_width = LaunchConfiguration('output_width')
    output_height = LaunchConfiguration('output_height')

    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{
            'left_optical_frame_name': 'left/image_rect',
            'right_optical_frame_name': 'right/image_rect',
            'module_id': module_id
        }],
    )

    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': output_width,
            'output_height': output_height,
            'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'left/camera_info'),
            ('image', 'left/image_raw'),
            ('resize/camera_info', 'left/camera_info_resize'),
            ('resize/image', 'left/image_resize')]
    )

    right_resize_node = ComposableNode(
        name='right_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': output_width,
            'output_height': output_height,
            'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'right/camera_info'),
            ('image', 'right/image_raw'),
            ('resize/camera_info', 'right/camera_info_resize'),
            ('resize/image', 'right/image_resize')]
    )

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': output_width,
            'output_height': output_height,
        }],
        remappings=[
            ('image_raw', 'left/image_resize'),
            ('camera_info', 'left/camera_info_resize'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect')
        ]
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': output_width,
            'output_height': output_height,
        }],
        remappings=[
            ('image_raw', 'right/image_resize'),
            ('camera_info', 'right/camera_info_resize'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold,
                     'input_layer_width': output_width,
                     'input_layer_height': output_height}],
        remappings=[
            ('left/camera_info', 'left/camera_info_rect'),
            ('right/camera_info', 'right/camera_info_rect')
        ]
    )

    point_cloud_node = ComposableNode(
        name='point_cloud_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'approximate_sync': False,
                'use_color': False,
                'use_system_default_qos': True,
        }],
        remappings=[
            ('left/image_rect_color', 'left/image_rect'),
            ('left/camera_info', 'left/camera_info_rect'),
            ('right/camera_info', 'right/camera_info_rect')
        ]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            argus_stereo_node, left_resize_node, right_resize_node,
            left_rectify_node, right_rectify_node,
            disparity_node, point_cloud_node
        ],
        output='screen'
    )

    return (launch.LaunchDescription(launch_args + [container]))
