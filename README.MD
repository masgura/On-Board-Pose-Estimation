# On-Board Pose Estimation

This repository contains the code to perform on-board pose estimation.

## Dependencies
- OpenCV >= 4.0.0
- C++ >= 17.0
- Cmake >= 3.18
- OpenVINO >= 2022.3.1

To install OpenVINO you can refer to https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_linux_header.html#doxid-openvino-docs-install-guides-installing-openvino-linux-header

To install OpenCV:

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip

mkdir -p build && cd build

cmake  ../opencv-4.x

cmake --build .
```

# Usage