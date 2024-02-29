# On-Board-Pose-Estimation

This repository contains the implementation of a satellite pose estimation software. The code is based on the YOLOv8 CPP Inference example found in https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-LibTorch-CPP-Inference

## Dependencies
- OpenCV >= 4.0.0
- C++ >= 17.0
- Cmake >= 3.18
- Libtorch >= 1.12.1

# Usage

The software is written in C++ and needs Libtorch and OpenCV. In Linux:

```
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip

mkdir -p build && cd build

cmake  ../opencv-4.x

cmake --build .
```
To create and launch the executable:

```
git clone https://github.com/masgura/On-Board-Pose-Estimation.git
cd On-Board-Pose-Estimation
mkdir build && cd build
cmake ..
make
./main
```

Please note that in the CMakeLists.txt it is needed to set the path of libtorch (usually "/home/<USER_NAME>/libtorch")
