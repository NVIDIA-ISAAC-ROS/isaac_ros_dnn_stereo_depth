# Instructions to Generate Disparity Maps for Stereo Images

These are instructions for generating a disparity map for a given stereo image pair.

In addition to supporting the ROS Bag input type, the `isaac_ros_ess_visualizer.py` script also supports raw input images and camera info files. Follow the steps to generate a disparity estimation from raw inputs:

1. Complete the [Quickstart](../README.md#quickstart) guide first.

2. Pull the example data:

   ```bash
   cd ~/workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity && \
   git lfs pull -X "" -I "resources/examples"
   ```

3. Launch the ESS Disparity Node:

   ```bash
   ros2 launch isaac_ros_ess isaac_ros_ess.launch.py engine_file_path:=/workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources/ess.engine
   ```

4. Visualize and validate the output of the package:

    ```bash
    ros2 run isaac_ros_ess isaac_ros_ess_visualizer.py --raw_inputs
    ```

    <div align="center"><img src="../resources/output_raw.png" width="600px" title="Output of ESS Disparity Node."/></div>

5. Try your own examples:

    ```bash
    ros2 run isaac_ros_ess isaac_ros_ess_visualizer.py --raw_inputs \
            --left_image_path '<Absolute path to your left image>' \
            --right_image_path '<Absolute path to your right image>' \
            --camera_info_path '<Absolute path your camera info json file>'
    ```
