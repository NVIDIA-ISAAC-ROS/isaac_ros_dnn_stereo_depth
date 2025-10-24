# Isaac ROS DNN Stereo Depth

NVIDIA-accelerated, deep learned stereo disparity estimation

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/ess3.0_conf0_r2b_storage_576p.gif/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_dnn_stereo_depth/ess3.0_conf0_r2b_storage_576p.gif/" width="800px"/></a></div>

---

## Webinar Available

Learn how to use this package by watching our on-demand webinar:
[Using ML Models in ROS 2 to Robustly Estimate Distance to Obstacles](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1407606/3998202/isaac-ros-webinar-series)

---

## Overview

Deep Neural Network (DNN)–based stereo models have become essential for depth estimation because they
overcome many of the fundamental limitations of classical and geometry-based stereo algorithms.

Traditional
stereo matching relies on explicitly finding pixel correspondences between left and right images using
handcrafted features. While effective in well-textured, ideal conditions, these approaches often fail in
“ill-posed” regions such as areas with reflections, specular highlights, texture-less surfaces, repetitive
patterns, occlusions, or even minor camera calibration errors. In such cases, classical algorithms may
produce incomplete or inaccurate depth maps, or be forced to discard information entirely, especially
when context-dependent filtering is not possible.

DNN-based stereo methods learn rich, hierarchical feature representations and context-aware
matching costs directly from data. These models
leverage semantic understanding and global scene context to infer depth, even in challenging environments
where traditional correspondence measures break down. Through training, DNNs can implicitly account for
real-world imperfections such as:

* calibration errors
* exposure differences
* hardware noise

Training increases DNN’s ability to
recognize and handle difficult regions like reflections or transparent surfaces. This results in more
robust, accurate, and dense depth predictions.

These advances are critical for robotics and autonomous
systems, enabling applications where both speed and accuracy
of depth perception are essential, such as:

* precise robotic arm manipulation
* reliable obstacle avoidance and navigation
* robust target tracking in dynamic or cluttered environments

DNN-based stereo methods consistently outperform classical techniques,
making them the preferred choice for modern depth perception tasks.

The superiority of DNN-based stereo methods is clearly demonstrated in the figure above where we compare
the output from a classical stereo algorithm, SGM, with DNN-based methods, ESS, and FoundationStereo.

SGM
produces a very noisy and error-prone disparity map, while ESS and FoundationStereo produce much smoother
and more accurate disparity maps. A closer look reveals that FoundationStereo produces the most accurate
map because it is better at handling  the plant in the distance and the railings on the left with smoother
estimates. Overall, you can see that FoundationStereo is better than ESS, and better than SGM, in terms of accuracy and quality.

DNN‐based stereo systems begin by passing the left and right images through shared
Convolutional backbones to extract multi‐scale feature maps that encode both texture and semantic
information. These feature maps are then compared across potential disparities by constructing
a learnable cost volume, which effectively represents the matching likelihood of each pixel at different
disparities. Successive 3D Convolutional (or 2D convolution + aggregation) stages then regularize and
refine this cost volume, integrating strong local cues—like edges and textures—and global scene
context—such as object shapes and layout priors—to resolve ambiguities. Finally, a soft‐argmax or
classification layer converts the refined cost volume into a dense disparity map, often followed by
lightweight refinement modules that enforce sub-pixel accuracy and respect learned priors (for example, smoothness
within objects, sharp transitions at boundaries), yielding a coherent estimate that gracefully handles
challenging scenarios where classical algorithms falter.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                       | Input Size<br/><br/>   | AGX Thor<br/><br/>                                                                                                                                                       | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DNN Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_ess_node.py)<br/><br/><br/>Full<br/><br/>          | 576p<br/><br/>         | [178 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_ess_node-agx_thor.json)<br/><br/><br/>22 ms @ 30Hz<br/><br/>        | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_ess_node-x86-5090.json)<br/><br/><br/>5.6 ms @ 30Hz<br/><br/>        |
| [DNN Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_light_ess_node.py)<br/><br/><br/>Light<br/><br/>   | 288p<br/><br/>         | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_light_ess_node-agx_thor.json)<br/><br/><br/>9.4 ms @ 30Hz<br/><br/> | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_light_ess_node-x86-5090.json)<br/><br/><br/>5.0 ms @ 30Hz<br/><br/>  |
| [DNN Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_ess_graph.py)<br/><br/><br/>Full<br/><br/>        | 576p<br/><br/>         | [73.6 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_ess_graph-agx_thor.json)<br/><br/><br/>29 ms @ 30Hz<br/><br/>      | [348 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_ess_graph-x86-5090.json)<br/><br/><br/>8.5 ms @ 30Hz<br/><br/>       |
| [DNN Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_ess_benchmark/scripts/isaac_ros_light_ess_graph.py)<br/><br/><br/>Light<br/><br/> | 288p<br/><br/>         | [219 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_light_ess_graph-agx_thor.json)<br/><br/><br/>17 ms @ 30Hz<br/><br/> | [350 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_light_ess_graph-x86-5090.json)<br/><br/><br/>7.3 ms @ 30Hz<br/><br/> |

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
* [`isaac_ros_foundationstereo`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_foundationstereo/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_foundationstereo/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_foundationstereo/index.html#try-more-examples)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_dnn_stereo_depth/isaac_ros_foundationstereo/index.html#api)

## Latest

Update 2025-10-24: Added FoundationStereo package
