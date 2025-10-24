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

"""
Proof-Of-Life test for the Isaac ROS FoundationStereo package.

    1. Sets up the complete FoundationStereo pipeline including:
       - Image preprocessing nodes for left and right images
       - Tensor processing nodes
       - TensorRT inference node
       - FoundationStereo decoder node
    2. Loads sample stereo images and publishes them
    3. Subscribes to the disparity output topic
    4. Verifies that the received disparity output has correct dimensions and encoding
"""

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion, MockModelGenerator
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage
import torch

MODEL_ONNX_PATH = '/tmp/foundationstereo_model.onnx'
MODEL_PLAN_PATH = '/tmp/foundationstereo_model.plan'
MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10
IMAGE_HEIGHT = 576
IMAGE_WIDTH = 960


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing FoundationStereo pipeline."""
    # Generate a dummy model with FoundationStereo-like I/O
    MockModelGenerator.generate(
        input_bindings=[
            MockModelGenerator.Binding('left', [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH], torch.float32),
            MockModelGenerator.Binding('right', [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH], torch.float32)
        ],
        output_bindings=[
            MockModelGenerator.Binding('disp', [1, 1, IMAGE_HEIGHT, IMAGE_WIDTH], torch.float32)
        ],
        output_onnx_path=MODEL_ONNX_PATH
    )

    # Left image preprocessing pipeline
    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'input_width': IMAGE_WIDTH,
            'input_height': IMAGE_HEIGHT,
            'output_width': IMAGE_WIDTH,
            'output_height': IMAGE_HEIGHT,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    )

    left_pad_node = ComposableNode(
        name='left_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'output_image_width': IMAGE_WIDTH,
            'output_image_height': IMAGE_HEIGHT,
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'input_tensor_shape': [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [3, IMAGE_HEIGHT, IMAGE_WIDTH],
            'output_tensor_shape': [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    )

    # Right image preprocessing pipeline (mirror of left pipeline)
    right_resize_node = ComposableNode(
        name='right_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'input_width': IMAGE_WIDTH,
            'input_height': IMAGE_HEIGHT,
            'output_width': IMAGE_WIDTH,
            'output_height': IMAGE_HEIGHT,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'right/image_rect'),
            ('camera_info', 'right/camera_info_rect'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    )

    right_pad_node = ComposableNode(
        name='right_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'output_image_width': IMAGE_WIDTH,
            'output_image_height': IMAGE_HEIGHT,
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'input_tensor_shape': [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [3, IMAGE_HEIGHT, IMAGE_WIDTH],
            'output_tensor_shape': [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]
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
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
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

    # TensorRT node
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'model_file_path': MODEL_ONNX_PATH,
            'engine_file_path': MODEL_PLAN_PATH,
            'input_tensor_names': ['left_image', 'right_image'],
            'input_binding_names': ['left', 'right'],
            'output_tensor_names': ['disparity'],
            'output_binding_names': ['disp'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    # FoundationStereo decoder node
    foundationstereo_decoder_node = ComposableNode(
        name='foundationstereo_decoder',
        package='isaac_ros_foundationstereo',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode',
        namespace=IsaacROSFoundationStereoTest.generate_namespace(),
        parameters=[{
            'disparity_tensor_name': 'disparity'
        }],
        remappings=[
            ('right/camera_info', 'right/camera_info_resize')
        ]
    )

    container = ComposableNodeContainer(
        name='foundationstereo_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            left_resize_node, left_pad_node, left_format_node, left_normalize_node,
            left_tensor_node, left_planar_node, left_reshape_node,
            right_resize_node, right_pad_node, right_format_node, right_normalize_node,
            right_tensor_node, right_planar_node, right_reshape_node,
            tensor_pair_sync_node, tensor_rt_node, foundationstereo_decoder_node
        ],
        output='screen'
    )

    return IsaacROSFoundationStereoTest.generate_test_description([container])


class IsaacROSFoundationStereoTest(IsaacROSBaseTest):
    """Validates that the FoundationStereo pipeline produces disparity outputs."""

    # filepath is required by IsaacROSBaseTest
    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    def _create_image(self):
        """Create a dummy image with specified dimensions."""
        image = Image()
        image.height = IMAGE_HEIGHT
        image.width = IMAGE_WIDTH
        image.encoding = 'rgb8'
        image.is_bigendian = False
        image.step = IMAGE_WIDTH * 3
        image.data = [0] * IMAGE_HEIGHT * IMAGE_WIDTH * 3
        return image

    @IsaacROSBaseTest.for_each_test_case()
    def test_stereo_disparity(self, test_folder):
        """Expect the node to produce disparity output given stereo images."""
        self.node._logger.info(f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(MODEL_PLAN_PATH):
            time_diff = time.time() - start_time
            if time_diff > MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            if time_diff > wait_cycles*10:
                self.node._logger.info(
                    f'Waiting for model generation to finish... ({time_diff:.0f}s passed)')
                wait_cycles += 1
            time.sleep(1)

        self.node._logger.info(
            f'Model generation was finished (took {(time.time() - start_time)}s)')

        received_messages = {}

        self.generate_namespace_lookup([
            'left/image_rect', 'right/image_rect',
            'left/camera_info_rect', 'right/camera_info_rect',
            'disparity'
        ])

        left_image_pub = self.node.create_publisher(
            Image, self.namespaces['left/image_rect'], self.DEFAULT_QOS)
        right_image_pub = self.node.create_publisher(
            Image, self.namespaces['right/image_rect'], self.DEFAULT_QOS)
        left_camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['left/camera_info_rect'], self.DEFAULT_QOS)
        right_camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['right/camera_info_rect'], self.DEFAULT_QOS)
        subs = self.create_logging_subscribers(
            [('disparity', DisparityImage)], received_messages)

        try:
            # Create dummy images
            left_image = self._create_image()
            right_image = self._create_image()
            # Load camera info from JSON
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                left_image_pub.publish(left_image)
                right_image_pub.publish(right_image)
                left_camera_info_pub.publish(camera_info)
                right_camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'disparity' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, "Didn't receive output on disparity topic!")

            # Verify disparity output properties
            disparity = received_messages['disparity']
            self.assertEqual(disparity.image.height, IMAGE_HEIGHT)
            self.assertEqual(disparity.image.width, IMAGE_WIDTH)
            self.assertEqual(disparity.image.encoding, '32FC1')
            self.assertEqual(disparity.image.step, disparity.image.width * 4)
            self.assertAlmostEqual(disparity.f, camera_info.p[0])
            self.assertAlmostEqual(disparity.t, -camera_info.p[3] / camera_info.p[0])
            self.assertAlmostEqual(disparity.min_disparity, 0.0)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(left_image_pub)
            self.node.destroy_publisher(right_image_pub)
            self.node.destroy_publisher(left_camera_info_pub)
            self.node.destroy_publisher(right_camera_info_pub)
