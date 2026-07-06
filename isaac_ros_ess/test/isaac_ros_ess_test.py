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

import os
import subprocess
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage

import tensorrt as trt


_TRT_VER = trt.__version__.replace('.', '_')
_ENGINE_FILE_PATH = f'/tmp/dummy_model_{_TRT_VER}.engine'


@pytest.mark.rostest
def generate_test_description():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    engine_file_path = _ENGINE_FILE_PATH
    trtexec_path = '/usr/src/tensorrt/bin/trtexec'
    if os.environ.get('TENSORRT_COMMAND', None):
        from python.runfiles import Runfiles
        _bazel_runfiles = Runfiles.Create()
        trtexec_path = _bazel_runfiles.Rlocation(os.environ['TENSORRT_COMMAND'])
    if not os.path.isfile(engine_file_path):
        args = [
            trtexec_path,
            f'--saveEngine={engine_file_path}',
            f'--onnx={dir_path}/dummy_model.onnx'
        ]
        print('Generating model engine file by command: ', ' '.join(args))
        result = subprocess.run(
            args,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise Exception(
                f'Failed to convert with status: {result.returncode}.\n'
                f'stderr:\n' + result.stderr.decode('utf-8')
            )

    namespace = IsaacROSDisparityTest.generate_namespace()
    model_width = IsaacROSDisparityTest.ESS_OUTPUT_WIDTH
    model_height = IsaacROSDisparityTest.ESS_OUTPUT_HEIGHT
    num_channels = 3

    left_format_node = ComposableNode(
        name='left_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=namespace,
        parameters=[{
            'image_width': IsaacROSDisparityTest.IMAGE_WIDTH,
            'image_height': IsaacROSDisparityTest.IMAGE_HEIGHT,
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
        namespace=namespace,
        parameters=[{
            'output_width': model_width,
            'output_height': model_height,
            'keep_aspect_ratio': False,
        }],
        remappings=[
            ('image', 'left/image_rgb'),
            ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    )
    left_normalize_node = ComposableNode(
        name='left_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        namespace=namespace,
        parameters=[{
            'mean': [127.5, 127.5, 127.5],
            'stddev': [127.5, 127.5, 127.5],
        }],
        remappings=[
            ('image', 'left/image_resize'),
            ('normalized_image', 'left/image_normalize')
        ]
    )
    left_tensor_node = ComposableNode(
        name='left_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=namespace,
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
        namespace=namespace,
        parameters=[{
            'input_tensor_shape': [model_height, model_width, num_channels],
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
        namespace=namespace,
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [num_channels, model_height, model_width],
            'output_tensor_shape': [1, num_channels, model_height, model_width]
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    )

    right_format_node = ComposableNode(
        name='right_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=namespace,
        parameters=[{
            'image_width': IsaacROSDisparityTest.IMAGE_WIDTH,
            'image_height': IsaacROSDisparityTest.IMAGE_HEIGHT,
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
        namespace=namespace,
        parameters=[{
            'output_width': model_width,
            'output_height': model_height,
            'keep_aspect_ratio': False,
        }],
        remappings=[
            ('image', 'right/image_rgb'),
            ('camera_info', 'right/camera_info_rect'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    )
    right_normalize_node = ComposableNode(
        name='right_normalize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        namespace=namespace,
        parameters=[{
            'mean': [127.5, 127.5, 127.5],
            'stddev': [127.5, 127.5, 127.5],
        }],
        remappings=[
            ('image', 'right/image_resize'),
            ('normalized_image', 'right/image_normalize')
        ]
    )
    right_tensor_node = ComposableNode(
        name='right_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=namespace,
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
        namespace=namespace,
        parameters=[{
            'input_tensor_shape': [model_height, model_width, num_channels],
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
        namespace=namespace,
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [num_channels, model_height, model_width],
            'output_tensor_shape': [1, num_channels, model_height, model_width]
        }],
        remappings=[
            ('tensor', 'right/tensor_planar'),
            ('reshaped_tensor', 'right/tensor_reshape')
        ]
    )

    tensor_pair_sync_node = ComposableNode(
        name='tensor_pair_sync_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        namespace=namespace,
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
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=namespace,
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['input_left', 'input_right'],
            'input_binding_names': ['input_left', 'input_right'],
            'output_tensor_names': ['output_left', 'output_conf'],
            'output_binding_names': ['output_left', 'output_conf'],
            'verbose': False,
            'force_engine_update': False,
        }]
    )

    decoder_node = ComposableNode(
        name='dnn_stereo_decoder',
        package='isaac_ros_dnn_stereo_decoder',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
        namespace=namespace,
        parameters=[{
            'disparity_tensor_name': 'output_left',
            'confidence_tensor_name': 'output_conf',
        }],
        remappings=[
            ('right/camera_info', 'right/camera_info_resize')
        ]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            left_format_node,
            left_resize_node,
            left_normalize_node,
            left_tensor_node,
            left_planar_node,
            left_reshape_node,
            right_format_node,
            right_resize_node,
            right_normalize_node,
            right_tensor_node,
            right_planar_node,
            right_reshape_node,
            tensor_pair_sync_node,
            tensor_rt_node,
            decoder_node,
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSDisparityTest.generate_test_description([container])


class IsaacROSDisparityTest(IsaacROSBaseTest):
    IMAGE_HEIGHT = 1080
    IMAGE_WIDTH = 1920
    # disparity output dimension fixed at 960x576
    ESS_OUTPUT_HEIGHT = 576
    ESS_OUTPUT_WIDTH = 960
    TIMEOUT = 50
    ENGINE_FILE_PATH = _ENGINE_FILE_PATH
    CAMERA_INFO_PATH = os.path.dirname(
        os.path.realpath(__file__)) + '/camera_info.json'

    def _create_image(self):
        image = Image()
        image.height = self.IMAGE_HEIGHT
        image.width = self.IMAGE_WIDTH
        image.encoding = 'rgb8'
        image.is_bigendian = False
        image.step = self.IMAGE_WIDTH * 3
        image.data = [0] * self.IMAGE_HEIGHT * self.IMAGE_WIDTH * 3
        return image

    def test_image_disparity(self):
        end_time = time.time() + self.TIMEOUT
        while time.time() < end_time:
            if os.path.isfile(self.ENGINE_FILE_PATH):
                break
        self.assertTrue(os.path.isfile(self.ENGINE_FILE_PATH),
                        'Model engine file was not generated in time.')

        received_messages = {}
        self.generate_namespace_lookup(['left/image_rect', 'right/image_rect',
                                        'left/camera_info_rect',
                                        'right/camera_info_rect',
                                        'disparity'])

        subs = self.create_logging_subscribers(
            [('disparity', DisparityImage)], received_messages)

        image_left_pub = self.node.create_publisher(
            Image, self.namespaces['left/image_rect'], self.DEFAULT_QOS
        )
        image_right_pub = self.node.create_publisher(
            Image, self.namespaces['right/image_rect'], self.DEFAULT_QOS
        )
        camera_info_left = self.node.create_publisher(
            CameraInfo, self.namespaces['left/camera_info_rect'], self.DEFAULT_QOS
        )
        camera_info_right = self.node.create_publisher(
            CameraInfo, self.namespaces['right/camera_info_rect'], self.DEFAULT_QOS
        )

        try:
            left_image = self._create_image()
            right_image = self._create_image()
            camera_info = JSONConversion.load_camera_info_from_json(
                self.CAMERA_INFO_PATH)

            end_time = time.time() + self.TIMEOUT
            done = False

            while time.time() < end_time:
                image_left_pub.publish(left_image)
                image_right_pub.publish(right_image)
                camera_info_left.publish(camera_info)
                camera_info_right.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'disparity' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Didnt recieve output on disparity topic')

            disparity = received_messages['disparity']
            self.assertEqual(disparity.image.height, self.ESS_OUTPUT_HEIGHT)
            self.assertEqual(disparity.image.width, self.ESS_OUTPUT_WIDTH)
            scaling_x = disparity.image.width / self.IMAGE_WIDTH
            scaling_y = disparity.image.height / self.IMAGE_HEIGHT
            min_scaling = min(scaling_x, scaling_y)
            self.assertEqual(disparity.image.encoding, '32FC1')
            self.assertEqual(disparity.image.step, disparity.image.width * 4)
            self.assertAlmostEqual(disparity.f, 434.9440002*min_scaling)
            self.assertAlmostEqual(disparity.t, -0.3678634)
            self.assertAlmostEqual(disparity.min_disparity, 0.0)
            self.assertAlmostEqual(disparity.max_disparity, 10000.0)

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(image_left_pub)
            self.node.destroy_publisher(image_right_pub)
            self.node.destroy_publisher(camera_info_right)
            self.node.destroy_publisher(camera_info_left)
