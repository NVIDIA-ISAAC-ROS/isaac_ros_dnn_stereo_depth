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
Proof-Of-Life test for the Isaac ROS DNNStereoDecoder node.

    1. Sets up a minimal stereo pipeline including:
       - Image preprocessing nodes for left and right images
       - Tensor processing nodes
       - TensorRT inference node producing a disparity tensor
       - DNNStereoDecoder node that converts disparity tensor to DisparityImage
    2. Publishes dummy stereo images and camera info
    3. Subscribes to the disparity output topic
    4. Verifies that the received disparity output has correct dimensions and encoding
"""

import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest, JSONConversion, MockModelGenerator
import launch
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage
import torch

MODEL_ONNX_PATH = '/tmp/dnn_stereo_decoder_model.onnx'
MODEL_PLAN_PATH = '/tmp/dnn_stereo_decoder_model.plan'
MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
MODEL_HEIGHT = 576
MODEL_WIDTH = 960


@pytest.mark.rostest
def generate_test_description() -> launch.LaunchDescription:
    """Generate launch description for testing DNNStereoDecoder pipeline."""
    # Generate a dummy model
    MockModelGenerator.generate(
        input_bindings=[
            MockModelGenerator.Binding(
                'left_image',
                [1, 3, MODEL_HEIGHT, MODEL_WIDTH],
                torch.float32),
            MockModelGenerator.Binding(
                'right_image',
                [1, 3, MODEL_HEIGHT, MODEL_WIDTH],
                torch.float32)
        ],
        output_bindings=[
            MockModelGenerator.Binding(
                'disparity',
                [1, 1, MODEL_HEIGHT, MODEL_WIDTH],
                torch.float32)
        ],
        output_onnx_path=MODEL_ONNX_PATH
    )

    # Include full pipeline launch under the test namespace
    pipeline_launch_path = os.path.join(
        get_package_share_directory('isaac_ros_dnn_stereo_decoder'),
        'launch',
        'isaac_ros_dnn_stereo_pipeline.launch.py'
    )
    pipeline_launch = GroupAction([
        PushRosNamespace(IsaacROSDNNStereoDecoderTest.generate_namespace()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pipeline_launch_path),
            launch_arguments={
                'image_width': str(IMAGE_WIDTH),
                'image_height': str(IMAGE_HEIGHT),
                'model_file_path': MODEL_ONNX_PATH,
                'engine_file_path': MODEL_PLAN_PATH,
                'min_disparity': '0.0',
                'max_disparity': '10000.0',
                'verbose': 'false',
                'force_engine_update': 'false'
            }.items()
        )
    ])

    return IsaacROSDNNStereoDecoderTest.generate_test_description([pipeline_launch])


class IsaacROSDNNStereoDecoderTest(IsaacROSBaseTest):
    """Validates that the DNNStereoDecoder pipeline produces disparity outputs."""

    # filepath is required by IsaacROSBaseTest
    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    def _create_image(self) -> Image:
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
    def test_stereo_disparity(self, test_folder: pathlib.Path) -> None:
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
            self.assertEqual(disparity.image.height, MODEL_HEIGHT)
            self.assertEqual(disparity.image.width, MODEL_WIDTH)
            self.assertEqual(disparity.image.encoding, '32FC1')
            self.assertEqual(disparity.image.step, disparity.image.width * 4)

            # The resize node scales the camera intrinsics from the
            # camera_info resolution down to the model input resolution.
            resize_scale = MODEL_WIDTH / camera_info.width
            self.assertAlmostEqual(disparity.f, camera_info.p[0] * resize_scale, places=2)
            # Baseline is invariant to resize (P[3] and P[0] scale by the same factor)
            self.assertAlmostEqual(disparity.t, -camera_info.p[3] / camera_info.p[0], places=2)
            self.assertAlmostEqual(disparity.min_disparity, 0.0)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(left_image_pub)
            self.node.destroy_publisher(right_image_pub)
            self.node.destroy_publisher(left_camera_info_pub)
            self.node.destroy_publisher(right_camera_info_pub)
