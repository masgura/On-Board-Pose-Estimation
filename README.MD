# On-Board Pose Estimation

This repository contains the code to perform on-board pose estimation.

## Dependencies
- OpenCV >= 4.0.0
- C++ >= 17.0
- Cmake >= 3.18
- OpenVINO >= 2024

To install OpenVINO you can refer to https://docs.openvino.ai/2024/get-started/install-openvino.html

To install OpenCV:

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip

mkdir -p build && cd build

cmake  ../opencv-4.x

cmake --build .
```
# Build

To build the project:

```
mkdir build && cd build
cmake ..
make
```

# Usage

This repository is based on the [ESA Kelvins Pose Estimation Challenge](https://kelvins.esa.int/satellite-pose-estimation-challenge/home/) and needs the test set from the [SPEED dataset](https://purl.stanford.edu/dz692fn7184). In particular, the code in [inference.cpp](https://github.com/masgura/On-board-pose-estimation/blob/main/src/inference.cpp) is used to process every image in the test set with two neural network:

1. YOLOv8 nano is used to detect the bounding box enclosing the Tango spacecraft.
2. The image zoomed on the Region of Interest (RoI) is then passed through a YOLOv8-pose nano model to regress the position of 11 keypoints.

Instead, [slab_score.cpp](https://github.com/masgura/On-board-pose-estimation/blob/main/src/slab_score.cpp) is used to estimate translation and orientation of the spacecraft through the EPnP algorithm, and to compute the SLAB score on the test set.
