#!/usr/bin/env python3

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
DNNStereo Disparity Visualizer.

By default, subscribes to existing ROS topics and visualizes/saves the disparity output.
Optionally loads images and camera info and publishes them when --raw_inputs is provided.
"""

# By default, visualizes disparity from existing ROS topics.
# If --raw_inputs is set, loads images/camera info and publishes them to the pipeline.

import argparse
import os

import cv2
import cv_bridge
from isaac_ros_test import JSONConversion
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage


ros_ws = os.environ['ISAAC_ROS_WS']


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the visualizer."""
    parser = argparse.ArgumentParser(description='DNN Stereo Disparity Visualizer')
    parser.add_argument('--save_image', action='store_true', help='Save output or display it.')
    parser.add_argument('--min_disp', type=int, default=0,
                        help='Min disparity for colormap normalization.')
    parser.add_argument('--max_disp', type=int, default=255,
                        help='Max disparity for colormap normalization.')
    parser.add_argument('--result_path', default=os.path.join(ros_ws, 'output.png'),
                        help='Absolute path to save your result.')
    parser.add_argument('--raw_inputs', action='store_true',
                        help='Use raw image and camera info files as inputs.')
    parser.add_argument('--left_image_path',
                        default=os.path.join(ros_ws, 'left.png'),
                        help='Absolute path to your left image.')
    parser.add_argument('--right_image_path',
                        default=os.path.join(ros_ws, 'right.png'),
                        help='Absolute path to your right image.')
    parser.add_argument('--camera_info_path',
                        default=os.path.join(ros_ws, 'camera.json'),
                        help='Absolute path to your camera info JSON file.')
    args = parser.parse_args()
    return args


class DnnStereoVisualizer(Node):
    """Visualize DNN stereo disparity; optionally publish inputs with --raw_inputs."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the visualizer node with CLI args."""
        super().__init__('dnn_stereo_visualizer')
        self.args = args
        self.encoding = 'rgb8'
        self._bridge = cv_bridge.CvBridge()

        self._disp_sub = self.create_subscription(
            DisparityImage, 'disparity', self.dnn_stereo_callback, 10)

        if self.args.raw_inputs:
            self._prepare_raw_inputs()

    def _prepare_raw_inputs(self) -> None:
        """Initialize publishers and prepare messages from provided raw inputs."""
        self._img_left_pub = self.create_publisher(
            Image, 'left/image_raw', 10)
        self._img_right_pub = self.create_publisher(
            Image, 'right/image_raw', 10)
        self._camera_left_pub = self.create_publisher(
            CameraInfo, 'left/camera_info', 10)
        self._camera_right_pub = self.create_publisher(
            CameraInfo, 'right/camera_info', 10)

        self.create_timer(5, self.timer_callback)

        if not os.path.isfile(self.args.left_image_path):
            self.get_logger().error(
                f'Left image file not found: {self.args.left_image_path}')
            raise FileNotFoundError(self.args.left_image_path)
        if not os.path.isfile(self.args.right_image_path):
            self.get_logger().error(
                f'Right image file not found: {self.args.right_image_path}')
            raise FileNotFoundError(self.args.right_image_path)

        left_img = cv2.imread(self.args.left_image_path)
        if left_img is None:
            self.get_logger().error(
                f'Failed to read left image: {self.args.left_image_path}')
            raise ValueError('Invalid left image file')
        right_img = cv2.imread(self.args.right_image_path)
        if right_img is None:
            self.get_logger().error(
                f'Failed to read right image: {self.args.right_image_path}')
            raise ValueError('Invalid right image file')

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        self.left_msg = self._bridge.cv2_to_imgmsg(left_img, self.encoding)
        self.right_msg = self._bridge.cv2_to_imgmsg(right_img, self.encoding)

        if not os.path.isfile(self.args.camera_info_path):
            self.get_logger().error(
                f'Camera info JSON not found: {self.args.camera_info_path}')
            raise FileNotFoundError(self.args.camera_info_path)
        try:
            self.camera_info = JSONConversion.load_camera_info_from_json(
                self.args.camera_info_path)
        except Exception as exc:
            self.get_logger().error(
                f'Failed to load camera info JSON: {self.args.camera_info_path} '
                f'({exc})'
            )
            raise

    def timer_callback(self) -> None:
        """Publish raw messages periodically when `--raw_inputs` is enabled."""
        self._img_left_pub.publish(self.left_msg)
        self._img_right_pub.publish(self.right_msg)
        self._camera_left_pub.publish(self.camera_info)
        self._camera_right_pub.publish(self.camera_info)
        self.get_logger().info('Inputs were published.')

    def dnn_stereo_callback(self, disp_msg: DisparityImage) -> None:
        """Handle disparity output, apply colormap and save or display it."""
        self.get_logger().info('Result was received.')
        disp_img = self._bridge.imgmsg_to_cv2(disp_msg.image)
        # Normalize and convert to colormap for visualization
        disp_img = (disp_img - self.args.min_disp) / (self.args.max_disp - self.args.min_disp)
        disp_img = np.clip(disp_img, 0., 1.)
        disp_img *= 255
        color_map = cv2.applyColorMap(disp_img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        if self.args.save_image:
            cv2.imwrite(self.args.result_path, color_map)
        else:
            cv2.imshow('dnn_stereo_output', color_map)
        cv2.waitKey(1)


def main() -> None:
    """Entrypoint for the DNNStereo visualizer."""
    args = get_args()
    rclpy.init()
    rclpy.spin(DnnStereoVisualizer(args))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
