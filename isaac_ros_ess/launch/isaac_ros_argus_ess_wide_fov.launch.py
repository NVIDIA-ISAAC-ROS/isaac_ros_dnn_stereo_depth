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

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

IMAGE_FRMAE_BBOX_X = 64
IMAGE_FRMAE_BBOX_Y = 184
IMAGE_FRMAE_BBOX_WIDTH = 1728
IMAGE_FRMAE_BBOX_HEIGHT = 768


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
            default_value='7',
            description='Index specifying the stereo camera module to use.'),
        DeclareLaunchArgument(
            'wide_fov',
            default_value='true',
            description='Flag to enable wide fov in argus camera.'),
    ]
    engine_file_path = LaunchConfiguration('engine_file_path')
    threshold = LaunchConfiguration('threshold')
    module_id = LaunchConfiguration('module_id')
    wide_fov = LaunchConfiguration('wide_fov')

    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{
            'left_optical_frame_name': 'left/image_rect',
            'right_optical_frame_name': 'right/image_rect',
            'module_id': module_id,
            'wide_fov': wide_fov,
            'type_negotiation_duration_s': 5,
        }],
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
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
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
            ('image_raw', 'right/image_raw'),
            ('camera_info', 'right/camera_info'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    left_crop_node = ComposableNode(
        name='left_crop',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::CropNode',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
            'crop_height': IMAGE_FRMAE_BBOX_HEIGHT,
            'crop_width': IMAGE_FRMAE_BBOX_WIDTH,
            'roi_top_left_x': IMAGE_FRMAE_BBOX_X,
            'roi_top_left_y': IMAGE_FRMAE_BBOX_Y,
            'crop_mode': 'BBOX',
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', 'left/camera_info_rect'),
            ('crop/image', 'left/image_crop'),
            ('crop/camera_info', 'left/camera_info_crop'),
        ]
    )

    right_crop_node = ComposableNode(
        name='right_crop',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::CropNode',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
            'crop_height': IMAGE_FRMAE_BBOX_HEIGHT,
            'crop_width': IMAGE_FRMAE_BBOX_WIDTH,
            'roi_top_left_x': IMAGE_FRMAE_BBOX_X,
            'roi_top_left_y': IMAGE_FRMAE_BBOX_Y,
            'crop_mode': 'BBOX',
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('image', 'right/image_rect'),
            ('camera_info', 'right/camera_info_rect'),
            ('crop/image', 'right/image_crop'),
            ('crop/camera_info', 'right/camera_info_crop'),
        ]
    )

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold,
                     'input_layer_width': 960,
                     'input_layer_height': 576,
                     'type_negotiation_duration_s': 5}],
        remappings=[
            ('left/image_rect', 'left/image_crop'),
            ('right/image_rect', 'right/image_crop'),
            ('left/camera_info', 'left/camera_info_crop'),
            ('right/camera_info', 'right/camera_info_crop'),
        ]
    )

    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
            'keep_aspect_ratio': False,
            'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('camera_info', 'left/camera_info_crop'),
            ('image', 'left/image_crop'),
            ('resize/camera_info', 'left/camera_info_resize'),
            ('resize/image', 'left/image_resize')]
    )

    point_cloud_node = ComposableNode(
        name='point_cloud_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'approximate_sync': False,
                'use_color': False,
                'use_system_default_qos': True,
                'type_negotiation_duration_s': 5,
        }],
        remappings=[
            ('left/image_rect_color', 'left/image_resize'),
            ('left/camera_info', 'left/camera_info_crop'),
            ('right/camera_info', 'right/camera_info_crop')
        ]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            argus_stereo_node,
            left_rectify_node,
            right_rectify_node,
            left_crop_node,
            right_crop_node,
            disparity_node,
            left_resize_node,
            point_cloud_node
        ],
        output='screen'
    )

    return (launch.LaunchDescription(launch_args + [container]))
