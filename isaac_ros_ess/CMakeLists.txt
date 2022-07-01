# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_ess LANGUAGES C CXX)


# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


# isaac_ros_ess_node
ament_auto_add_library(isaac_ros_ess_node SHARED src/ess_disparity_node.cpp)
target_compile_definitions(isaac_ros_ess_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(isaac_ros_ess_node)
rclcpp_components_register_node(isaac_ros_ess_node
  PLUGIN "nvidia::isaac_ros::dnn_stereo_disparity::ESSDisparityNode"
  EXECUTABLE isaac_ros_ess)

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install package executable
install(TARGETS isaac_ros_ess_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install extensions directory
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  install(DIRECTORY gxf/lib/gxf_x86_64/ DESTINATION share/${PROJECT_NAME}/gxf)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  install(DIRECTORY gxf/lib/gxf_jetpack50/ DESTINATION share/${PROJECT_NAME}/gxf)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_ess_test.py TIMEOUT "300")
endif()

# Launch/Visualizer python scripts
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  launch/isaac_ros_ess.launch.py
  launch/isaac_ros_argus_ess.launch.py
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/isaac_ros_ess_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()
