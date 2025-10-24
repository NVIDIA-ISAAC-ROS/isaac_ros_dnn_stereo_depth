# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MODEL_NUM_CHANNELS = 3  # RGB channels

SIM_IMAGE_WIDTH = 1280
SIM_IMAGE_HEIGHT = 800

ISAAC_ROS_ASSETS_PATH = os.path.join(os.environ['ISAAC_ROS_WS'], 'isaac_ros_assets')
ISAAC_ROS_MODELS_PATH = os.path.join(ISAAC_ROS_ASSETS_PATH, 'models')
FOUNDATION_STEREO_MODELS_PATH = os.path.join(ISAAC_ROS_MODELS_PATH, 'foundationstereo')
FOUNDATION_STEREO_ENGINE_PATH = os.path.join(FOUNDATION_STEREO_MODELS_PATH,
                                             'foundationstereo.engine')


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            default_value=FOUNDATION_STEREO_ENGINE_PATH,
            description='The absolute file path to the TensorRT engine file'),

        DeclareLaunchArgument(
            'input_image_width',
            default_value=str(SIM_IMAGE_WIDTH),
            description='The input image width'),

        DeclareLaunchArgument(
            'input_image_height',
            default_value=str(SIM_IMAGE_HEIGHT),
            description='The input image height'),

        DeclareLaunchArgument(
            'model_input_width',
            default_value='960',
            description='The model input width'),

        DeclareLaunchArgument(
            'model_input_height',
            default_value='576',
            description='The model input height'),

        DeclareLaunchArgument(
            'container_name',
            default_value='foundationstereo_container',
            description='Name for ComposableNodeContainer'),
    ]

    # Image preprocessing parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    model_input_width = LaunchConfiguration('model_input_width')
    model_input_height = LaunchConfiguration('model_input_height')
    engine_file_path = LaunchConfiguration('engine_file_path')
    container_name = LaunchConfiguration('container_name')

    # Left image preprocessing nodes
    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': input_image_width,
            'input_height': input_image_height,
            'output_width': model_input_width,
            'output_height': model_input_height,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'front_stereo_camera/left/image_rect_color'),
            ('camera_info', 'front_stereo_camera/left/camera_info'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    )

    left_pad_node = ComposableNode(
        name='left_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': model_input_width,
            'output_image_height': model_input_height,
            'border_type': 'REPLICATE'
        }],
        remappings=[
            ('image', 'left/image_resize'),
            ('padded_image', 'left/image_pad'),
        ]
    )

    left_format_node = ComposableNode(
        name='left_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': model_input_width,
            'image_height': model_input_height,
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'left/image_pad'),
            ('image', 'left/image_rgb')
        ]
    )

    left_normalize_node = ComposableNode(
        name='left_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
        }],
        remappings=[
            ('image', 'left/image_rgb'),
            ('normalized_image', 'left/image_normalize')
        ]
    )

    left_tensor_node = ComposableNode(
        name='left_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'left_image',
        }],
        remappings=[
            ('image', 'left/image_normalize'),
            ('tensor', 'left/tensor'),
        ]
    )

    left_planar_node = ComposableNode(
        name='left_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [model_input_height, model_input_width, MODEL_NUM_CHANNELS],
            'output_tensor_name': 'left_image'
        }],
        remappings=[
            ('interleaved_tensor', 'left/tensor'),
            ('planar_tensor', 'left/tensor_planar')
        ]
    )

    left_reshape_node = ComposableNode(
        name='left_reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, model_input_height, model_input_width],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, model_input_height, model_input_width]
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    )

    # Right image preprocessing nodes
    right_resize_node = ComposableNode(
        name='right_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': input_image_width,
            'input_height': input_image_height,
            'output_width': model_input_width,
            'output_height': model_input_height,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'front_stereo_camera/right/image_rect_color'),
            ('camera_info', 'front_stereo_camera/right/camera_info'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    )

    right_pad_node = ComposableNode(
        name='right_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': model_input_width,
            'output_image_height': model_input_height,
            'border_type': 'REPLICATE'
        }],
        remappings=[
            ('image', 'right/image_resize'),
            ('padded_image', 'right/image_pad'),
        ]
    )

    right_format_node = ComposableNode(
        name='right_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': model_input_width,
            'image_height': model_input_height,
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'right/image_pad'),
            ('image', 'right/image_rgb')
        ]
    )

    right_normalize_node = ComposableNode(
        name='right_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
        }],
        remappings=[
            ('image', 'right/image_rgb'),
            ('normalized_image', 'right/image_normalize')
        ]
    )

    right_tensor_node = ComposableNode(
        name='right_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'right_image',
        }],
        remappings=[
            ('image', 'right/image_normalize'),
            ('tensor', 'right/tensor'),
        ]
    )

    right_planar_node = ComposableNode(
        name='right_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [model_input_height, model_input_width, MODEL_NUM_CHANNELS],
            'output_tensor_name': 'right_image'
        }],
        remappings=[
            ('interleaved_tensor', 'right/tensor'),
            ('planar_tensor', 'right/tensor_planar')
        ]
    )

    right_reshape_node = ComposableNode(
        name='right_reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, model_input_height, model_input_width],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, model_input_height, model_input_width]
        }],
        remappings=[
            ('tensor', 'right/tensor_planar'),
            ('reshaped_tensor', 'right/tensor_reshape')
        ]
    )

    # Tensor sync node
    tensor_pair_sync_node = ComposableNode(
        name='tensor_pair_sync_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{
            'input_tensor1_name': 'left_image',
            'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'left_image',
            'output_tensor2_name': 'right_image'
        }],
        remappings=[
            ('tensor1', 'left/tensor_reshape'),
            ('tensor2', 'right/tensor_reshape'),
        ]
    )

    # TensorRT node for stereo processing
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['left_image', 'right_image'],
            'input_binding_names': ['left_image', 'right_image'],
            'output_tensor_names': ['disparity'],
            'output_binding_names': ['disparity'],
            'force_engine_update': False
        }]
    )

    # Disparity decoder node
    foundationstereo_decoder_node = ComposableNode(
        name='foundationstereo_decoder',
        package='isaac_ros_foundationstereo',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'disparity'
        }],
        remappings=[
            ('right/camera_info', 'right/camera_info_resize')
        ]
    )

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
            'use_color': True,
            'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'left/image_rgb'),
                    ('left/camera_info', 'left/camera_info_resize'),
                    ('right/camera_info', 'right/camera_info_resize')])

    foundationstereo_container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Left preprocessing pipeline
            left_resize_node, left_pad_node, left_format_node,
            left_normalize_node, left_tensor_node, left_planar_node, left_reshape_node,
            # Right preprocessing pipeline
            right_resize_node, right_pad_node, right_format_node,
            right_normalize_node, right_tensor_node, right_planar_node, right_reshape_node,
            # Preprocessing, inference and decoding
            tensor_pair_sync_node, tensor_rt_node, foundationstereo_decoder_node,
            # Pointcloud node
            pointcloud_node
        ],
        output='screen'
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_foundationstereo'), 'config', 'isaac_ros_foundationstereo_isaac_sim.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')

    return launch.LaunchDescription(launch_args + [foundationstereo_container, rviz_node])
