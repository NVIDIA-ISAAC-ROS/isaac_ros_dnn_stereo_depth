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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    camera = LaunchConfiguration('camera')
    camera_arg = DeclareLaunchArgument(
        'camera',
        description='camera',
    )

    engine_file_path = LaunchConfiguration('engine_file_path')
    engine_file_path_arg = DeclareLaunchArgument(
        'engine_file_path',
        default_value='',
        description='The absolute path to the ESS engine plan.')

    threshold = LaunchConfiguration('threshold')
    threshold_arg = DeclareLaunchArgument(
        'threshold',
        default_value='0.0',
        description='Threshold value ranges between 0.0 and 1.0 '
                    'for filtering disparity with confidence.')
    launch_args = [
        camera_arg,
        engine_file_path_arg,
        threshold_arg,
    ]

    left_image_topic = PythonExpression(["'/' + '", camera, "' + '/left/image_compressed'"])
    left_info_topic = PythonExpression(["'/' + '", camera, "' + '/left/camera_info'"])
    right_image_topic = PythonExpression(["'/' + '", camera, "' + '/right/image_compressed'"])
    right_info_topic = PythonExpression(["'/' + '", camera, "' + '/right/camera_info'"])
    left_raw_image_topic = PythonExpression(["'/' + '", camera, "' + '/left/image_raw'"])
    right_raw_image_topic = PythonExpression(["'/' + '", camera, "' + '/right/image_raw'"])

    left_decoder = ComposableNode(
        name='left_decoder',
        package='isaac_ros_h264_decoder',
        plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
        namespace=camera,
        remappings=[
            ('image_compressed', left_image_topic),
            ('image_uncompressed', left_raw_image_topic),
        ],
    )

    right_decoder = ComposableNode(
        name='right_decoder',
        package='isaac_ros_h264_decoder',
        plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
        namespace=camera,
        remappings=[
            ('image_compressed', right_image_topic),
            ('image_uncompressed', right_raw_image_topic),
        ],
    )

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 1920,
            'output_height': 1200,
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('image_raw', left_raw_image_topic),
            ('camera_info', left_info_topic),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect')
        ]
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 1920,
            'output_height': 1200,
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('image_raw', right_raw_image_topic),
            ('camera_info', right_info_topic),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'threshold': threshold,
            'input_layer_width': 960,
            'input_layer_height': 576,
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('left/image_rect', 'left/image_rect'),
            ('right/image_rect', 'right/image_rect'),
            ('left/camera_info', 'left/camera_info_rect'),
            ('right/camera_info', 'right/camera_info_rect'),
        ]
    )

    disparity_to_depth_node = ComposableNode(
        name='DisparityToDepthNode',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
    )

    container = ComposableNodeContainer(
        name='depth_container',
        namespace='depth',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            left_decoder,
            right_decoder,
            left_rectify_node,
            right_rectify_node,
            disparity_node,
            disparity_to_depth_node,
        ],
        output='screen'
    )

    return (LaunchDescription(launch_args + [container]))
