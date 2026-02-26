# Pallet Detector

## Overview

ROS packages for pallet detector. Includes:

- **pallet_camera_gz**: Gazebo simulation with an RGBD camera on a stand and a pallet model
- **pallet_detector**: ROS package to detect pallets from point cloud data

## Prerequisites

- ROS Noetic 
- Gazebo 

## How to Build

```bash
cd ~/ros_docker_ws
catkin_make
source devel/setup.bash
```

## How to Run

To launch the Gazebo simulation with the camera stand and pallet:

```bash
roslaunch pallet_camera_gz pallet_camera.launch
```

1. Start Gazebo with the `pallet_camera.world`
2. Spawn the camera stand with an RGBD depth sensor
3. Publish TF frames (`world` → `stand_link` → `rgbd_camera_link`)

### Available Topics

| Topic | Description |
|---|---|
| `/demo/rgb/image_raw` | RGB camera image |
| `/demo/rgb/camera_info` | RGB camera intrinsics |
| `/demo/depth/image_raw` | Depth image |
| `/demo/depth/camera_info` | Depth camera intrinsics |
| `/demo/depth/points` | 3D point cloud |

### Visualize 

In Rviz add displays for:
- **Image** → topic: `/demo/rgb/image_raw`
- **PointCloud2** → topic: `/demo/depth/points` (set Fixed Frame to `world`)

