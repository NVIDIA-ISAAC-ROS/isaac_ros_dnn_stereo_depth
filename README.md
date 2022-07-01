# Isaac ROS DNN Stereo Disparity
DNN Stereo Disparity includes packages for predicting disparity of stereo input.

<div align="center"><img src="resources/warehouse.gif" width="500px" title="Applying Colormap of ESS Disparity Node Ouput to the Input Image."/></div>

## Overview
This repository provides an NVIDIA hardware-accelerated package for DNN-based stereo disparity. Stereo disparity (with additional processing) can produce a depth image or point cloud of a scene for robot navigation. The `isaac_ros_ess` package uses the [ESS](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity) DNN to perform stereo depth estimation via continuous disparity prediction. Given a pair of stereo input images, the package generates a disparity map of the left input image.

### ESS DNN
[ESS](https://arxiv.org/pdf/1803.09719.pdf) stands for Enhanced Semi-Supervised stereo disparity DNN, which was developed by NVIDIA. The ESS DNN is used to predict the disparity for each pixel from stereo camera image pairs. This network has improvements over classic CV approaches that use epi-polar geometry to compute disparity, as the DNN can learn to predict disparity in cases where epi-polar geometry feature matching fails. The semi-supervised learning and stereo disparity matching makes the DNN robust to environment that is unseen in the training datasets and occluded objects. This DNN is optimized for and evaluated with RGB global shutter stereo camera images, and accuracy may vary with monochrome images.

The predicted [disparity](https://en.wikipedia.org/wiki/Binocular_disparity) values represent the distance a point moves from one image to the other in a stereo image pair (a.k.a. the binocular image pair). The disparity is inversely proportional to the depth (i.e. `disparity = focalLength x baseline / depth`). Given the [focal length](https://en.wikipedia.org/wiki/Focal_length) and [baseline](https://en.wikipedia.org/wiki/Stereo_camera) of the camera that generates a stereo image pair, the predicted disparity map from the `isaac_ros_ess` package can be used to compute depth and generate a [point cloud](https://en.wikipedia.org/wiki/Point_cloud).

### Isaac ROS NITROS Acceleration
This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance
The following are the benchmark performance results of the prepared pipelines in this package, by supported platform:

| Pipeline                   | AGX Orin | AGX Xavier | x86_64 w/ RTX 3060 Ti |
| -------------------------- | -------- | ---------- | --------------------- |
| ESS Disparity Node (1080p) | 51 fps   | 24 fps     | 118 fps               |

## Table of Contents
- [Isaac ROS DNN Stereo Disparity](#isaac-ros-dnn-stereo-disparity)
  - [Overview](#overview)
    - [ESS DNN](#ess-dnn)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try Advanced Examples](#try-advanced-examples)
    - [Try NITROS-Accelerated Graph with Argus](#try-nitros-accelerated-graph-with-argus)
    - [Use Different Models](#use-different-models)
    - [Customize Your Dev Environment](#customize-your-dev-environment)
  - [Package Reference](#package-reference)
    - [`isaac_ros_ess`](#isaac_ros_ess)
      - [Overview](#overview-1)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
      - [Input Restrictions](#input-restrictions)
      - [Output Interpretations](#output-interpretations)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [Deep Learning Troubleshooting](#deep-learning-troubleshooting)
    - [Package not found while launching the visualizer script](#package-not-found-while-launching-the-visualizer-script)
      - [Symptom](#symptom)
      - [Solution](#solution)
    - [Problem reserving CacheChange in reader](#problem-reserving-cachechange-in-reader)
      - [Symptom](#symptom-1)
      - [Solution](#solution-1)
  - [Updates](#updates)

## Latest Update
Update 2022-06-30: Initial release.

## Supported Platforms
This package is designed and tested to be compatible with ROS2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.


| Platform | Hardware                                                                                                                                                                                          | Software                                                                                                             | Notes                                                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [AGX Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)<br/>[AGX Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.0.1 DP](https://developer.nvidia.com/embedded/jetpack)                                                    | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                        | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.6.1+](https://developer.nvidia.com/cuda-downloads) |

### Docker
To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note:** All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart
1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).

2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`:
   ```bash
   mkdir -p ~/workspaces/isaac_ros-dev/src && cd ~/workspaces/isaac_ros-dev/src
   ```
   ```bash
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
   ```
   ```bash
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
   ```
   ```bash
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
   ```
   ```bash
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_disparity
   ```

3. Pull down a ROS Bag of sample data:
   ```bash
   cd ~/workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity && \
   git lfs pull -X "" -I "resources/rosbags/ess_rosbag"
   ```

4. Launch the Docker container using the `run_dev.sh` script:
   ```bash
   cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh
   ```

5. Download the pre-trained ESS model from the [ESS model page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity):
   ```bash
   cd /workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources && \
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity/versions/1.0.0/files/ess.etlt'
   ```

6. Convert the encrypted model (`.etlt`) to a TensorRT engine plan:
   ```bash
   /opt/nvidia/tao/tao-converter -k ess -t fp16 -e /workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources/ess.engine -o output_left /workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources/ess.etlt
   ```

7. Build and source the workspace:
   ```bash
   cd /workspaces/isaac_ros-dev && \
     colcon build && \
     source install/setup.bash
   ```
   **Note**: We recommend rebuilding the package each time the source files are edited. Before rebuilding, first clean the workspace by running `rm -r build install log`.

8. (Optional) Run tests to verify complete and correct installation:
   ```bash
   colcon test --executor sequential
   ```

9. Launch the ESS Disparity Node:
   ```bash
   ros2 launch isaac_ros_ess isaac_ros_ess.launch.py engine_file_path:=/workspaces/isaac_ros-dev/src/isaac_ros_dnn_stereo_disparity/resources/ess.engine
   ```

10. Visualize and validate the output of the package:
    ```bash
    ros2 run isaac_ros_ess isaac_ros_ess_visualizer.py
    ```
    <div align="center"><img src="resources/output_rosbag.png" width="500px" title="Output of ESS Disparity Node."/></div>

## Next Steps
### Try Advanced Examples
To continue exploring the DNN Stereo Disparity package, check out the following suggested examples:

- [Generating disparity maps from a stereo pair of image files](./docs/visualize-image.md)

### Try NITROS-Accelerated Graph with Argus
If you have an Argus-compatible camera, you can launch the NITROS-accelerated graph by following the [tutorial](docs/tutorial-nitros-graph.md).

### Use Different Models
Click [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/model-preparation.md) for more information about how to use NGC models.

### Customize Your Dev Environment
To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/modify-dockerfile.md).

## Package Reference
### `isaac_ros_ess`
#### Overview
The `isaac_ros_ess` package offers functionality to generate a stereo disparity map from stereo images using a trained ESS model. Given a pair of stereo input images, the package generates a continuous disparity image for the left input image.

#### Usage
```bash
ros2 launch isaac_ros_ess isaac_ros_ess.launch.py engine_file_path:=<your ESS engine plan absolute path>
```
#### ROS Parameters
| ROS Parameter      | Type     | Default        | Description                                                                                                                                                                                                         |
| ------------------ | -------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `engine_file_path` | `string` | N/A - Required | The absolute path to the ESS engine file.                                                                                                                                                                           |
| `image_type`       | `string` | `"RGB_U8"`.    | The input image encoding type. Supports `"RGB_U8"` and `"BGR_U8"`. <br> Note that if users do not specify this parameter and there is an upstreaming Isaac ROS NITROS node, the type will be decided automatically. |

#### ROS Topics Subscribed
| ROS Topic           | Interface                                                                                                      | Description                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `left/image_rect`   | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg)           | The left image of a stereo pair.  |
| `right/image_rect`  | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg)           | The right image of a stereo pair. |
| `left/camera_info`  | [sensor_msgs/CameraInfo](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg) | The left camera model.            |
| `right/camera_info` | [sensor_msgs/CameraInfo](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg) | The right camera model.           |

#### ROS Topics Published
| ROS Topic   | Interface                                                                                                              | Description                                 |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `disparity` | [stereo_msgs/DisparityImage](https://github.com/ros2/common_interfaces/blob/humble/stereo_msgs/msg/DisparityImage.msg) | The continuous stereo disparity estimation. |

#### Input Restrictions
1. The input left and right images must have the **same dimension and resolution**, and the resolution must be **no larger than `1920x1200`**.

2. Each input pair (`left/image_rect`, `right/image_rect`, `left/camera_info` and `right/camera_info`) should have the **same timestamp**; otherwise, the synchronizing module inside the ESS Disparity Node will drop the input with smaller timestamps.

#### Output Interpretations
1. The `isaas_ros_ess` package outputs a disparity image with the same resolution as the input stereo pairs. The input images are rescaled to the ESS model input size before inferencing and the output prediction is rescaled back before publishing. To alter this behavior, use input images with the model-native resolution: `W=960, H=576`.

2. The left and right `CameraInfo` are used to composite a `stereo_msgs/DisparityImage`. If you only care about the disparity image, and don't need the baseline and focal length information, you can pass dummy camera messages.

## Troubleshooting
### Isaac ROS Troubleshooting
For solutions to problems with Isaac ROS, check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

### Deep Learning Troubleshooting
For solutions to problems using DNN models, check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/troubleshooting.md).

### Package not found while launching the visualizer script
#### Symptom
```bash
$ ros2 run isaac_ros_ess isaac_ros_ess_visualizer.py
Package 'isaac_ros_ess' not found
```

#### Solution
Use the `colcon build --packages-up-to isaac_ros_ess` command to build `isaac_ros_ess`; do not use the `--symlink-install` option. Run `source install/setup.bash` after the build.

### Problem reserving CacheChange in reader
#### Symptom
When using a ROS bag as input, `isaac_ros_ess` throws an error if the input topics are published too fast:
```bash
[component_container-1] 2022-06-24 09:04:43.584 [RTPS_MSG_IN Error] (ID:281473268431152) Problem reserving CacheChange in reader: 01.0f.cd.10.ab.f2.65.b6.01.00.00.00|0.0.20.4 -> Function processDataMsg
```

#### Solution
Make sure that the ROS bag has a reasonable size and publish rate.

## Updates

| Date       | Changes         |
| ---------- | --------------- |
| 2022-06-30 | Initial release |
