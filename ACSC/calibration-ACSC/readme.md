# Lidar & Camera Calibration



## 1. Dependency

- PCL
- python 3.X
- opencv-python
- open3d
- transforms3d
- pyyaml
- mayavi

## 2. Preparation

```
cd /path/to/your/pcd-calibration/segmentation
mkdir build
cd build
cmake ..
msbuild segmentationext.vcxproj
copy Debug/segmentationext.pyd into /path/to/your/pcd-calibration/seg


```



## 3. Extrinsic Calibration

### 3.1 Data format

The images and LiDAR point clouds data format in the data folder:

```
│  distortion
│  intrinsic
│
├─images
│      000000.png
│      000001.png
│      ......
│
├─pcds
│      000000.npy
│      000001.npy
│      ......
│
└─ROIs
        000000.txt
        000001.txt
        ......
```

Among them, the `images` directory contains images containing checkerboard at different placements and angles.

The `pcds` directory  contains point clouds corresponding to the images with pcd format, or you can change the load function in the code. Data in each pcd  hasthe shape of `N x 7`, and each row is the `x`, `y`, `z` , ...., and `reflectance` information of the point;

### 3.2 Data collection

......



### 3.3 Camera intrinsic parameters

There are many tools for camera intrinsic calibration, here we recommend using the [Camera Calibrator App](https://www.mathworks.com/help/vision/ug/single-camera-calibrator-app.html) in MATLAB, or the [Camera Calibration Tools](http://wiki.ros.org/camera_calibration) in ROS, to calibrate the camera intrinsic.

camera intrinsic matrix:

```
fx s x0
0 fy y0
0 0  1
```

 camera distortion vector:

```
k1  k2  p1  p2  k3
```

### 3.4  Extrinsic Calibration

Modify the calibration configuration file in directory `config`, sample.yaml for example:

1. Modify the `root` under `data`
2. Modify the `chessboard` parameter under `data`, change `W` and `H` to the number of inner corners of the checkerboard that you use (note that, it is **not the number of squares, but the number of inner corners**.  Modify `GRID_SIZE` to the side length of a single little white / black square of the checkerboard (unit is m);