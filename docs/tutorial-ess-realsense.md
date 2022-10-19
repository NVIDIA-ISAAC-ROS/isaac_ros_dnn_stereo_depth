# Tutorial for DNN Stereo Depth Estimation using a RealSense camera

<div align="center"><img src="../resources/realsense_example.png"/></div>

## Overview

This tutorial demonstrates how to perform stereo-camera-based reconstruction using a [RealSense](https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html) camera and [isaac_ros_ess](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_disparity).

> Note: This tutorial has been tested with a RealSense D455/D435 camera connected to a Jetson Xavier AGX, as well as an x86 PC with an NVIDIA graphics card.

## Tutorial Walkthrough

1. Complete the [realsense setup tutorial](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md).
2. Complete steps 1-2 described in the [Quickstart Guide](../README.md#quickstart).
3. Open a new terminal and launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

4. Build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

5. Run the launch file, which launches the example and waits for 10 seconds:

    ```bash
    ros2 launch isaac_ros_ess isaac_ros_ess_realsense.launch.py engine_file_path:=/workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources/ess.engine
    ```
