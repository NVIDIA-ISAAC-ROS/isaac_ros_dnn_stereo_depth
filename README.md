# Isaac ROS DNN Stereo Depth

NVIDIA-accelerated, deep learned stereo disparity estimation

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/ess3.0_conf0_r2b_storage_576p.gif/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/ess3.0_conf0_r2b_storage_576p.gif/" width="800px"/></a></div>

---

## Webinar Available

Learn how to use this package by watching our on-demand webinar:
[Using ML Models in ROS 2 to Robustly Estimate Distance to Obstacles](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

## Overview

The vision depth perception problem is generally useful in many fields of robotics such as estimating
the pose of a robotic arm in an object manipulation task, estimating distance of static or moving targets
in autonomous robot navigation, tracking targets in delivery robots and so on.
[Isaac ROS DNN Stereo Depth](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_depth) is targeted at two Isaac applications,
Isaac Manipulator and Isaac Perceptor. In Isaac Manipulator application, ESS is deployed in
[Isaac ROS cuMotion](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html)
package as a plug-in node to provide depth perception maps for robot arm motion planning and control.
In this scenario, multi-camera stereo streams of industrial robot arms on a table task are passed to ESS to
obtain corresponding depth streams. The depth streams are used to segment the relative distance of robot arms from
corresponding objects on the table; thus providing signals for collision avoidance and fine-grain control.
Similarly, the Isaac Perceptor application uses several Isaac ROS packages, namely,
[Isaac ROS Nova](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nova/index.html),
[Isaac ROS Visual Slam](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/index.html),
[Isaac ROS Stereo Depth (ESS)](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/index.html),
[Isaac ROS Nvblox](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/index.html)
and [Isaac ROS Image Pipeline](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/index.html).

ESS is deployed in Isaac Perceptor to enable Nvblox to create
3D voxelized images of the robot surroundings. Specifically, the Nova developer suite provides 3x stereo-camera
streams to Isaac Perceptor. Each stream corresponds to the front, left, and right cameras.
In both Isaac Manipulator and Isaac Perceptor, a camera-specific image processing pipeline consisting of
GPU-accelerated operations, provides rectification and undistortion of the input stereo images. All stereo stream image
pair are time synchronized before before passing them to ESS. ESS node outputs corresponding depth maps for all three
preprocessed image streams and combines the depth images with motion signals provided by cuVSLAM module.
The combined depth and motion integrated signals are fed to Nvblox module to produce a dense 3D volumetric scene
reconstruction of the surrounding scene.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess_nodegraph.png/" width="800px"/></a></div>

Above, ESS node is used in a graph of nodes to provide a disparity prediction from an input left and right stereo image pair.
The rectify and resize nodes pre-process the left and right frames to the appropriate resolution.
The aspect ratio of the image is recommended to be maintained to avoid degrading the depth output quality.
The graph for DNN encode, DNN inference, and DNN decode is included in the ESS node. Inference is performed using
TensorRT, as the ESS DNN model is designed with optimizations supported by TensorRT.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                                | Orin NX<br/><br/>                                                                                                                                                | x86_64 w/ RTX 4090<br/><br/>                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DNN Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_ess_node.py)<br/><br/><br/>Full<br/><br/>          | 576p<br/><br/><br/><br/> | [103 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_node-agx_orin.json)<br/><br/><br/>12 ms @ 30Hz<br/><br/>        | [42.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_node-orin_nx.json)<br/><br/><br/>26 ms @ 30Hz<br/><br/>       | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_node-x86-4090.json)<br/><br/><br/>2.3 ms @ 30Hz<br/><br/>        |
| [DNN Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_light_ess_node.py)<br/><br/><br/>Light<br/><br/>   | 288p<br/><br/><br/><br/> | [306 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_node-agx_orin.json)<br/><br/><br/>5.6 ms @ 30Hz<br/><br/> | [143 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_node-orin_nx.json)<br/><br/><br/>9.4 ms @ 30Hz<br/><br/> | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_node-x86-4090.json)<br/><br/><br/>1.6 ms @ 30Hz<br/><br/>  |
| [DNN Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_ess_graph.py)<br/><br/><br/>Full<br/><br/>        | 576p<br/><br/><br/><br/> | [33.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_graph-agx_orin.json)<br/><br/><br/>25 ms @ 30Hz<br/><br/>      | [35.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_graph-orin_nx.json)<br/><br/><br/>34 ms @ 30Hz<br/><br/>      | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_graph-x86-4090.json)<br/><br/><br/>5.6 ms @ 30Hz<br/><br/>       |
| [DNN Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_light_ess_graph.py)<br/><br/><br/>Light<br/><br/> | 288p<br/><br/><br/><br/> | [179 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_graph-agx_orin.json)<br/><br/><br/>14 ms @ 30Hz<br/><br/> | [126 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_graph-orin_nx.json)<br/><br/><br/>15 ms @ 30Hz<br/><br/> | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_graph-x86-4090.json)<br/><br/><br/>4.4 ms @ 30Hz<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_ess`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess/index.html#api)

## Latest

Update 2024-09-26: Updated for ESS 4.1 trained on additional samples
