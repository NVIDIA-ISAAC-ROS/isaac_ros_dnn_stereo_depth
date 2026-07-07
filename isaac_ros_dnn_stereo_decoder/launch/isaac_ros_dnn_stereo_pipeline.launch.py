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
Launch a full DNN Stereo Disparity pipeline.

This pipeline performs:
- Image preprocessing for left/right images (format conversion, resize, pad, normalize)
- Tensor preparation (image to tensor, interleaved to planar, reshape)
- Tensor pair synchronization
- TensorRT inference producing a disparity tensor
- DNNStereoDecoder to convert disparity tensor into a `NitrosDisparityImage`

Expected launch arguments:
- image_width: input/output width used by the model
- image_height: input/output height used by the model
- model_file_path: path to ONNX model
- engine_file_path: path to TensorRT engine
- min_disparity: minimum valid disparity value (inclusive)
- max_disparity: maximum valid disparity value (inclusive)
- verbose: enable TensorRT verbose logging
- force_engine_update: rebuild engine if true
"""

from typing import List

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> launch.LaunchDescription:
    """Create and return the stereo disparity pipeline launch description."""
    # Exposed parameters
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    model_input_width = LaunchConfiguration('model_input_width')
    model_input_height = LaunchConfiguration('model_input_height')
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    min_disparity = LaunchConfiguration('min_disparity')
    max_disparity = LaunchConfiguration('max_disparity')
    verbose = LaunchConfiguration('verbose')
    force_engine_update = LaunchConfiguration('force_engine_update')

    # Left preprocessing
    left_format_node = ComposableNode(
        name='left_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': image_width,
            'image_height': image_height,
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'left/image_rect'),
            ('image', 'left/image_rgb')
        ]
    )
    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': image_width,
            'input_height': image_height,
            'output_width': model_input_width,
            'output_height': model_input_height,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'left/image_rgb'),
            ('camera_info', 'left/camera_info_rect'),
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
    left_normalize_node = ComposableNode(
        name='left_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
        }],
        remappings=[
            ('image', 'left/image_pad'),
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
            'input_tensor_shape': [model_input_height, model_input_width, 3],
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
            'input_tensor_shape': [3, model_input_height, model_input_width],
            'output_tensor_shape': [1, 3, model_input_height, model_input_width]
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    )

    # Right preprocessing
    right_format_node = ComposableNode(
        name='right_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': image_width,
            'image_height': image_height,
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'right/image_rect'),
            ('image', 'right/image_rgb')
        ]
    )
    right_resize_node = ComposableNode(
        name='right_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': image_width,
            'input_height': image_height,
            'output_width': model_input_width,
            'output_height': model_input_height,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'right/image_rgb'),
            ('camera_info', 'right/camera_info_rect'),
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
    right_normalize_node = ComposableNode(
        name='right_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
        }],
        remappings=[
            ('image', 'right/image_pad'),
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
            'input_tensor_shape': [model_input_height, model_input_width, 3],
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
            'input_tensor_shape': [3, model_input_height, model_input_width],
            'output_tensor_shape': [1, 3, model_input_height, model_input_width]
        }],
        remappings=[
            ('tensor', 'right/tensor_planar'),
            ('reshaped_tensor', 'right/tensor_reshape')
        ]
    )

    # Synchronize tensors
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

    # TensorRT inference
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['left_image', 'right_image'],
            'input_binding_names': ['left_image', 'right_image'],
            'output_tensor_names': ['disparity'],
            'output_binding_names': ['disparity'],
            'verbose': verbose,
            'force_engine_update': force_engine_update
        }]
    )

    # Decoder
    dnn_stereo_decoder_node = ComposableNode(
        name='dnn_stereo_decoder',
        package='isaac_ros_dnn_stereo_decoder',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'disparity',
            'min_disparity': min_disparity,
            'max_disparity': max_disparity
        }],
        remappings=[
            # Feed the resized right camera info into the decoder
            ('right/camera_info', 'right/camera_info_resize')
        ]
    )

    nodes: List[ComposableNode] = [
        left_format_node, left_resize_node, left_pad_node, left_normalize_node,
        left_tensor_node, left_planar_node, left_reshape_node,
        right_format_node, right_resize_node, right_pad_node, right_normalize_node,
        right_tensor_node, right_planar_node, right_reshape_node,
        tensor_pair_sync_node, tensor_rt_node, dnn_stereo_decoder_node
    ]

    container = ComposableNodeContainer(
        package='rclcpp_components',
        name='dnn_stereo_decoder_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=nodes,
        output='screen'
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument('image_width', default_value='960'),
        DeclareLaunchArgument('image_height', default_value='576'),
        DeclareLaunchArgument('model_input_width', default_value='960'),
        DeclareLaunchArgument('model_input_height', default_value='576'),
        DeclareLaunchArgument(
            'model_file_path',
            default_value='/tmp/dnn_stereo_decoder_model.onnx'
        ),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='/tmp/dnn_stereo_decoder_model.plan'
        ),
        DeclareLaunchArgument('min_disparity', default_value='0.0'),
        DeclareLaunchArgument('max_disparity', default_value='10000.0'),
        DeclareLaunchArgument('verbose', default_value='false'),
        DeclareLaunchArgument('force_engine_update', default_value='false'),
        container
    ])
