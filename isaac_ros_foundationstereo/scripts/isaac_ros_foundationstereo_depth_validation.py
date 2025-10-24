#!/usr/bin/env python3

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

# This script plays a rosbag and listens for depth frame from FoundationStereo depth
# pipeline then validates the depth of the white board in the rosbag frames.

import argparse
import subprocess

import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

DEPTH_METER = 1.99
DEPTH_THRESHOLD_METER = 0.01
NUMBER_OF_FRAMES = 100


def get_args():
    parser = argparse.ArgumentParser(description='FoundationStereo Depth Validation')
    parser.add_argument('--topic', default='depth/depth', help='Depth topic.')
    parser.add_argument('--depth_nominal',
                        default=DEPTH_METER,
                        help='Test depth nominal in meter.')
    parser.add_argument('--depth_threshold',
                        default=DEPTH_THRESHOLD_METER,
                        help='Test threshold in meter.')
    parser.add_argument('--nframes',
                        default=NUMBER_OF_FRAMES,
                        help='Number of frames for the test.')
    parser.add_argument('--rosbag',
                        default='/workspaces/isaac_ros-dev/ros_ws/src/'
                                'isaac_ros_dnn_stereo_depth/resources/'
                                'rosbags/hawk_2m',
                        help='Absolute path to the rosbag.')
    parser.add_argument('--replay_rate',
                        default='1.0',
                        help='Rosbag replay Rate.')
    parser.add_argument('--camera',
                        default='front_stereo_camera',
                        help='Camera used for the depth validation.')
    args = parser.parse_args()

    return args


class DepthValidation(Node):

    def __init__(self, args):
        super().__init__('foundationstereo_depth_validation')

        # Parsing parameters
        self.args = args
        self.nframes = int(args.nframes)
        self.threshold = float(args.depth_threshold)
        self.depth = float(args.depth_nominal)
        self.count = 0
        self.fail_count = 0
        self.depth_avg = []
        self.get_logger().info('Validation with depth={} threshold={}'
                               .format(self.depth, self.threshold))

        self.bridge = cv_bridge.CvBridge()
        self._depth_sub = self.create_subscription(Image, args.topic,
                                                   self.depth_callback, 10)
        self.play_rosbag()

    def play_rosbag(self):
        left_image_topic = '/' + self.args.camera + '/left/image_compressed '
        left_info_topic = '/' + self.args.camera + '/left/camera_info '
        right_image_topic = '/' + self.args.camera + '/right/image_compressed '
        right_info_topic = '/' + self.args.camera + '/right/camera_info '

        self.rosbag_process = subprocess.Popen(
            'ros2 bag play ' + self.args.rosbag +
            ' --rate ' + self.args.replay_rate +
            ' --wait-for-all-acked 0 ' +
            ' --topics ' + left_image_topic + left_info_topic +
            right_image_topic + right_info_topic +
            ' --disable-keyboard-controls ',
            shell=True)

    def depth_callback(self, msg):
        if (self.count % 10 == 0):
            self.get_logger().info('{} frames were received.'
                                   .format(self.count))
        depth_img = self.bridge.imgmsg_to_cv2(msg)

        # Take average of depth with range xxx
        data = np.flip(depth_img, 0)
        depth_board = data[400:550, 420:550]
        depth_avg = np.average(depth_board)
        self.depth_avg.append(round(depth_avg, 4))

        if depth_avg > self.depth + self.threshold:
            self.get_logger().info('Depth validation failed with depth'
                                   ' bigger than {:.4f}'
                                   .format(self.depth + self.threshold))
            self.fail_count += 1
        if depth_avg < self.depth - self.threshold:
            self.get_logger().info('Depth validation failed with depth less'
                                   ' than {:.4f}'
                                   .format(self.depth - self.threshold))
            self.fail_count += 1
        self.count += 1

        # Test for NUMBER_OF_FRAMES
        if self.count >= self.nframes:
            depth_max = np.max(self.depth_avg)
            depth_min = np.min(self.depth_avg)
            if self.fail_count == 0:
                self.get_logger().info('Depth validation passed for {} frames '
                                       'in range {:.4f} - {:.4f}.'
                                       .format(self.nframes, depth_min,
                                               depth_max))
            else:
                self.get_logger().info('Depth validation failed for {}  out of'
                                       ' {} frames in range {:.4f} - {:.4f}.'
                                       .format(self.fail_count, self.nframes,
                                               depth_min, depth_max))

            self.get_logger().info('Shutting down')
            self.rosbag_process.kill()
            raise SystemExit


def main():
    args = get_args()
    rclpy.init()

    try:
        rclpy.spin(DepthValidation(args))
    except SystemExit:
        rclpy.logging.get_logger('foundationstereo_depth_validation').info('Done')


if __name__ == '__main__':
    main()
