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

MODEL_NUM_CHANNELS = 3  # RGB channels


class IsaacROSEssLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        # Image preprocessing parameters
        input_image_width = LaunchConfiguration('input_image_width')
        input_image_height = LaunchConfiguration('input_image_height')
        model_input_width = LaunchConfiguration('model_input_width')
        model_input_height = LaunchConfiguration('model_input_height')

        # TensorRT parameters
        model_file_path = LaunchConfiguration('model_file_path')
        engine_file_path = LaunchConfiguration('engine_file_path')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        verbose = LaunchConfiguration('verbose')
        force_engine_update = LaunchConfiguration('force_engine_update')
        ess_plugin_file_path = LaunchConfiguration('ess_plugin_file_path')

        # Decoder parameters
        min_disparity = LaunchConfiguration('min_disparity')
        max_disparity = LaunchConfiguration('max_disparity')
        threshold = LaunchConfiguration('threshold')

        return {
            # Left image preprocessing pipeline
            'left_format_node': ComposableNode(
                name='left_format_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{
                    'image_width': input_image_width,
                    'image_height': input_image_height,
                    'encoding_desired': 'rgb8',
                }],
                remappings=[
                    ('image_raw', 'left/image_rect'),
                    ('image', 'left/image_rgb')
                ]
            ),
            'left_resize_node': ComposableNode(
                name='left_resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': model_input_width,
                    'output_height': model_input_height,
                    'keep_aspect_ratio': False,
                }],
                remappings=[
                    ('image', 'left/image_rgb'),
                    ('camera_info', 'left/camera_info_rect'),
                    ('resize/image', 'left/image_resize'),
                    ('resize/camera_info', 'left/camera_info_resize'),
                ]
            ),
            'left_normalize_node': ComposableNode(
                name='left_normalize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
                parameters=[{
                    'mean': [127.5, 127.5, 127.5],
                    'stddev': [127.5, 127.5, 127.5],
                }],
                remappings=[
                    ('image', 'left/image_resize'),
                    ('normalized_image', 'left/image_normalize')
                ]
            ),
            'left_tensor_node': ComposableNode(
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
            ),
            'left_planar_node': ComposableNode(
                name='left_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{
                    'input_tensor_shape': [model_input_height, model_input_width,
                                           MODEL_NUM_CHANNELS],
                    'output_tensor_name': 'left_image'
                }],
                remappings=[
                    ('interleaved_tensor', 'left/tensor'),
                    ('planar_tensor', 'left/tensor_planar')
                ]
            ),
            'left_reshape_node': ComposableNode(
                name='left_reshape_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[{
                    'output_tensor_name': 'left_image',
                    'input_tensor_shape': [MODEL_NUM_CHANNELS, model_input_height,
                                           model_input_width],
                    'output_tensor_shape': [1, MODEL_NUM_CHANNELS, model_input_height,
                                            model_input_width]
                }],
                remappings=[
                    ('tensor', 'left/tensor_planar'),
                    ('reshaped_tensor', 'left/tensor_reshape')
                ]
            ),

            # Right image preprocessing pipeline
            'right_format_node': ComposableNode(
                name='right_format_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{
                    'image_width': input_image_width,
                    'image_height': input_image_height,
                    'encoding_desired': 'rgb8',
                }],
                remappings=[
                    ('image_raw', 'right/image_rect'),
                    ('image', 'right/image_rgb')
                ]
            ),
            'right_resize_node': ComposableNode(
                name='right_resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': model_input_width,
                    'output_height': model_input_height,
                    'keep_aspect_ratio': False,
                }],
                remappings=[
                    ('image', 'right/image_rgb'),
                    ('camera_info', 'right/camera_info_rect'),
                    ('resize/image', 'right/image_resize'),
                    ('resize/camera_info', 'right/camera_info_resize'),
                ]
            ),
            'right_normalize_node': ComposableNode(
                name='right_normalize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
                parameters=[{
                    'mean': [127.5, 127.5, 127.5],
                    'stddev': [127.5, 127.5, 127.5],
                }],
                remappings=[
                    ('image', 'right/image_resize'),
                    ('normalized_image', 'right/image_normalize')
                ]
            ),
            'right_tensor_node': ComposableNode(
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
            ),
            'right_planar_node': ComposableNode(
                name='right_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{
                    'input_tensor_shape': [model_input_height, model_input_width,
                                           MODEL_NUM_CHANNELS],
                    'output_tensor_name': 'right_image'
                }],
                remappings=[
                    ('interleaved_tensor', 'right/tensor'),
                    ('planar_tensor', 'right/tensor_planar')
                ]
            ),
            'right_reshape_node': ComposableNode(
                name='right_reshape_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[{
                    'output_tensor_name': 'right_image',
                    'input_tensor_shape': [MODEL_NUM_CHANNELS, model_input_height,
                                           model_input_width],
                    'output_tensor_shape': [1, MODEL_NUM_CHANNELS, model_input_height,
                                            model_input_width]
                }],
                remappings=[
                    ('tensor', 'right/tensor_planar'),
                    ('reshaped_tensor', 'right/tensor_reshape')
                ]
            ),

            # Tensor sync node
            'tensor_pair_sync_node': ComposableNode(
                name='tensor_pair_sync_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
                parameters=[{
                    'input_tensor1_name': 'left_image',
                    'input_tensor2_name': 'right_image',
                    'output_tensor1_name': 'input_left',
                    'output_tensor2_name': 'input_right'
                }],
                remappings=[
                    ('tensor1', 'left/tensor_reshape'),
                    ('tensor2', 'right/tensor_reshape'),
                ]
            ),

            # TensorRT node for stereo processing
            'tensor_rt_node': ComposableNode(
                name='tensor_rt',
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                parameters=[{
                    'model_file_path': model_file_path,
                    'engine_file_path': engine_file_path,
                    'input_tensor_names': input_tensor_names,
                    'input_binding_names': input_binding_names,
                    'output_tensor_names': output_tensor_names,
                    'output_binding_names': output_binding_names,
                    'verbose': verbose,
                    'force_engine_update': force_engine_update,
                    'custom_plugin_lib': ess_plugin_file_path
                }]
            ),

            # Disparity decoder node
            'ess_decoder_node': ComposableNode(
                name='dnn_stereo_decoder',
                package='isaac_ros_dnn_stereo_decoder',
                plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
                parameters=[{
                    'disparity_tensor_name': 'output_left',
                    'confidence_tensor_name': 'output_conf',
                    'confidence_threshold': threshold,
                    'min_disparity': min_disparity,
                    'max_disparity': max_disparity,
                }],
                remappings=[
                    ('right/camera_info', 'right/camera_info_resize')
                ]
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
            'model_file_path': DeclareLaunchArgument(
                'model_file_path',
                default_value='',
                description='The absolute file path to the ONNX file'
            ),
            'engine_file_path': DeclareLaunchArgument(
                'engine_file_path',
                default_value='',
                description='The absolute file path to the TensorRT engine file'
            ),
            'ess_plugin_file_path': DeclareLaunchArgument(
                'ess_plugin_file_path',
                default_value='',
                description='The absolute file path to the ESS TensorRT plugin library (.so)'
            ),
            'input_image_width': DeclareLaunchArgument(
                'input_image_width',
                default_value='960',
                description='The input image width'
            ),
            'input_image_height': DeclareLaunchArgument(
                'input_image_height',
                default_value='576',
                description='The input image height'
            ),
            'model_input_width': DeclareLaunchArgument(
                'model_input_width',
                default_value='960',
                description='The model input width'
            ),
            'model_input_height': DeclareLaunchArgument(
                'model_input_height',
                default_value='576',
                description='The model input height'
            ),
            'input_tensor_names': DeclareLaunchArgument(
                'input_tensor_names',
                default_value='["input_left", "input_right"]',
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value='["input_left", "input_right"]',
                description='A list of input tensor binding names (specified by model)'
            ),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["output_left", "output_conf"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["output_left", "output_conf"]',
                description='A list of output tensor binding names (specified by model)'
            ),
            'verbose': DeclareLaunchArgument(
                'verbose',
                default_value='False',
                description='Whether TensorRT should verbosely log or not'
            ),
            'force_engine_update': DeclareLaunchArgument(
                'force_engine_update',
                default_value='False',
                description='Whether TensorRT should update the TensorRT engine file or not'
            ),
            'threshold': DeclareLaunchArgument(
                'threshold',
                default_value='0.4'
            ),
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
