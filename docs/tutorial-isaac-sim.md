# Tutorial for DNN Stereo Depth Estimation with Isaac Sim

<div align="center"><img src="../resources/Rviz.png" width="800px"/></div>

## Overview

This tutorial walks you through a pipeline to [estimate depth](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_disparity) with stereo images from Isaac Sim.

## Tutorial Walkthrough

1. Complete steps 1-7 listed in the [Quickstart section](../README.md#quickstart) of the main README.
2. Install and launch Isaac Sim following the steps in the [Isaac ROS Isaac Sim Setup Guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-sil-setup.md)
3. Open the Isaac ROS Common USD scene (using the **Content** window) located at:

   `omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd`.

   Wait for the scene to load completely.
   > **Note:** To use a different server, replace `localhost` with `<your_nucleus_server>`
4. Go to the **Stage** tab and select `/World/Carter_ROS/ROS_Cameras/ros2_create_camera_right_info`. In the **Property** tab, locate the **Compute Node -> Inputs -> stereoOffset -> X** value and change it from `0` to `-175.92`.
    <div align="center"><img src="../resources/Isaac_sim_set_stereo_offset.png" width="500px"/></div>

5. Enable the right camera for a stereo image pair. Go to the **Stage** tab and select `/World/Carter_ROS/ROS_Cameras/enable_camera_right`, then tick the **Condition** checkbox.
    <div align="center"><img src="../resources/Isaac_sim_enable_stereo.png" width="500px"/></div>
6. Press **Play** to start publishing data from the Isaac Sim application.
    <div align="center"><img src="../resources/Isaac_sim_play.png" width="800px"/></div>
7. Open a second terminal and attach to the container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```

8. In the second terminal, start the `isaac_ros_ess` node using the launch files:

    ```bash
    ros2 launch isaac_ros_ess isaac_ros_ess_isaac_sim.launch.py engine_file_path:=/workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources/ess.engine
    ```

9. Optionally, you can run the visualizer script to visualize the disparity image.

    ```bash
    ros2 run isaac_ros_ess isaac_ros_ess_visualizer.py
    ```

    <div align="center"><img src="../resources/Visualizer_isaac_sim.png" width="500px"/></div>
