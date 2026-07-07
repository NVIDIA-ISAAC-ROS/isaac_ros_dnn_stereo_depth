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
Correctness test to ensure one output per left/right image pair.

This launches the same DNNStereoDecoder pipeline and publishes multiple
timestamp-synchronized left/right images and camera infos. It verifies that
for each input pair timestamp, a corresponding DisparityImage is produced.
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
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
MODEL_HEIGHT = 576
MODEL_WIDTH = 960


@pytest.mark.rostest
def generate_test_description() -> launch.LaunchDescription:
    """Generate launch description for testing DNNStereoDecoder pipeline."""
    # Generate a dummy TensorRT model
    MockModelGenerator.generate(
        input_bindings=[
            MockModelGenerator.Binding(
                'left_image',
                [1, 3, MODEL_HEIGHT, MODEL_WIDTH],
                torch.float32
            ),
            MockModelGenerator.Binding(
                'right_image',
                [1, 3, MODEL_HEIGHT, MODEL_WIDTH],
                torch.float32
            )
        ],
        output_bindings=[
            MockModelGenerator.Binding(
                'disparity',
                [1, 1, MODEL_HEIGHT, MODEL_WIDTH],
                torch.float32
            )
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
        PushRosNamespace(IsaacROSDNNStereoDecoderOutputPerPairTest.generate_namespace()),
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

    return IsaacROSDNNStereoDecoderOutputPerPairTest.generate_test_description([pipeline_launch])


class IsaacROSDNNStereoDecoderOutputPerPairTest(IsaacROSBaseTest):
    """Validate that each synchronized left/right pair yields one disparity output."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    def _create_image(self) -> Image:
        image = Image()
        image.height = IMAGE_HEIGHT
        image.width = IMAGE_WIDTH
        image.encoding = 'rgb8'
        image.is_bigendian = False
        image.step = IMAGE_WIDTH * 3
        image.data = [0] * IMAGE_HEIGHT * IMAGE_WIDTH * 3
        return image

    @IsaacROSBaseTest.for_each_test_case()
    def test_output_per_image_pair(self, test_folder: pathlib.Path) -> None:
        # Wait for model engine generation
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(MODEL_PLAN_PATH):
            time_diff = time.time() - start_time
            if time_diff > MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            if time_diff > wait_cycles * 10:
                self.node._logger.info(
                    f'Waiting for model generation to finish... ({time_diff:.0f}s passed)')
                wait_cycles += 1
            time.sleep(1)

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

        disparity_sub = self.create_logging_subscribers(
            subscription_requests=[('disparity', DisparityImage)],
            received_messages=received_messages,
            accept_multiple_messages=True
        )

        try:
            left_image = self._create_image()
            right_image = self._create_image()
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            NUM_PAIRS = 10
            TOTAL_TIMEOUT = 1000
            PER_PAIR_TIMEOUT = 100
            end_time = time.time() + TOTAL_TIMEOUT

            produced_for_timestamps = set()
            last_checked_index = 0

            def _stamp_key(stamp):
                return (stamp.sec, stamp.nanosec)

            # Pair 0 is a warmup: republish continuously until the pipeline
            # responds, so that subscriber discovery and TensorRT init are
            # complete before we start the single-publish reliability checks.
            WARMUP_PAIRS = 1

            for ctr in range(NUM_PAIRS):
                if time.time() > end_time:
                    self.fail('Timeout before publishing all image pairs')

                timestamp = self.node.get_clock().now().to_msg()
                left_image.header.stamp = timestamp
                right_image.header.stamp = timestamp
                camera_info.header.stamp = timestamp

                is_warmup = ctr < WARMUP_PAIRS

                if not is_warmup:
                    left_image_pub.publish(left_image)
                    right_image_pub.publish(right_image)
                    left_camera_info_pub.publish(camera_info)
                    right_camera_info_pub.publish(camera_info)

                pair_deadline = time.time() + PER_PAIR_TIMEOUT
                pair_done = False

                while time.time() < pair_deadline:
                    if is_warmup:
                        left_image_pub.publish(left_image)
                        right_image_pub.publish(right_image)
                        left_camera_info_pub.publish(camera_info)
                        right_camera_info_pub.publish(camera_info)

                    rclpy.spin_once(self.node, timeout_sec=0.1)

                    if 'disparity' in received_messages:
                        msgs = received_messages['disparity']
                        for j in range(last_checked_index, len(msgs)):
                            msg = msgs[j]
                            stamp = msg.header.stamp
                            if (stamp.sec == 0 and stamp.nanosec == 0 and
                                    msg.image.header is not None):
                                stamp = msg.image.header.stamp
                            produced_for_timestamps.add(_stamp_key(stamp))
                        last_checked_index = len(msgs)

                    if _stamp_key(timestamp) in produced_for_timestamps:
                        pair_done = True
                        break

                self.assertTrue(
                    pair_done,
                    'No disparity output produced for the left/right pair '
                    f'with timestamp {timestamp.sec}.{timestamp.nanosec}'
                )

            self.assertGreaterEqual(
                len(received_messages.get('disparity', [])),
                NUM_PAIRS,
                'Total disparity outputs fewer than number of input pairs'
            )

        finally:
            self.node.destroy_subscription(disparity_sub)
            self.node.destroy_publisher(left_image_pub)
            self.node.destroy_publisher(right_image_pub)
            self.node.destroy_publisher(left_camera_info_pub)
            self.node.destroy_publisher(right_camera_info_pub)
