# Isaac ROS DNN Stereo Depth

Hardware-accelerated, deep learned stereo disparity estimation

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/ess3.0_conf0_r2b_storage_576p.gif/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/ess3.0_conf0_r2b_storage_576p.gif/" width="800px"/></a></div>

---

## Webinar Available

Learn how to use this package by watching our on-demand webinar:
[Using ML Models in ROS 2 to Robustly Estimate Distance to Obstacles](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

## Overview

[Isaac ROS DNN Stereo Depth](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_depth) provides a GPU-accelerated package for DNN-based
stereo disparity. Stereo disparity is calculated from a
time-synchronized image pair sourced from a stereo camera and is used to
produce a depth image or a point cloud for a scene. The `isaac_ros_ess`
package uses the [ESS DNN
model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity)
to perform stereo depth estimation via continuous disparity prediction.
Given a pair of stereo input images, the package generates a disparity
map of the left input image.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_ess_nodegraph.png/" width="800px"/></a></div>

ESS is used in a graph of nodes to provide a disparity prediction from an input left and right stereo image pair.
Images to ESS need to be rectified and resized to the appropriate input resolution.
The aspect ratio of the image is recommended to be maintained, so the image may need to be cropped and resized to maintain the input aspect ratio.
The graph for DNN encode, DNN inference, and DNN decode is included in the ESS node.
Inference is performed using TensorRT, as the ESS DNN model is designed with optimizations supported by TensorRT.
ESS node is agnostic to the model dimension and disparity output has the same dimension as the ESS model.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                             | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                         | Orin NX<br/><br/>                                                                                                                                         | x86_64 w/ RTX 4060 Ti<br/><br/>                                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DNN Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_ess_node.py)<br/><br/><br/>Full<br/><br/>          | 576p<br/><br/><br/><br/> | [78.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_node-agx_orin.json)<br/><br/><br/>4.1 ms<br/><br/>      | [27.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_node-orin_nx.json)<br/><br/><br/>6.2 ms<br/><br/>      | [204 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_node-nuc_4060ti.json)<br/><br/><br/>4.4 ms<br/><br/>       |
| [DNN Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_light_ess_node.py)<br/><br/><br/>Light<br/><br/>   | 288p<br/><br/><br/><br/> | [288 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_node-agx_orin.json)<br/><br/><br/>5.6 ms<br/><br/> | [128 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_node-orin_nx.json)<br/><br/><br/>5.6 ms<br/><br/> | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_node-nuc_4060ti.json)<br/><br/><br/>4.2 ms<br/><br/> |
| [DNN Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_ess_graph.py)<br/><br/><br/>Full<br/><br/>        | 576p<br/><br/><br/><br/> | [74.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_graph-agx_orin.json)<br/><br/><br/>20 ms<br/><br/>      | [26.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_graph-orin_nx.json)<br/><br/><br/>42 ms<br/><br/>      | [191 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_ess_graph-nuc_4060ti.json)<br/><br/><br/>11 ms<br/><br/>       |
| [DNN Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts/isaac_ros_light_ess_graph.py)<br/><br/><br/>Light<br/><br/> | 288p<br/><br/><br/><br/> | [260 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_graph-agx_orin.json)<br/><br/><br/>13 ms<br/><br/> | [116 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_graph-orin_nx.json)<br/><br/><br/>16 ms<br/><br/> | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_light_ess_graph-nuc_4060ti.json)<br/><br/><br/>12 ms<br/><br/> |

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

Update 2023-10-18: Updated for ESS 3.0 with confidence thresholding in multiple resolutions.
