cmake_minimum_required(VERSION 3.10)
project(OnBoardPoseEstimation)

set(CMAKE_CXX_STANDARD 17)

# -------------- JSON --------------
find_package(nlohmann_json REQUIRED)

# -------------- OpenCV --------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- OpenVINO --------------
set(OpenVINO_DIR "/opt/intel/openvino_2024.1.0/runtime/cmake")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# -------------- YOLOv8 --------------
set(YOLOV8_SOURCES src/YOLOv8/yolov8.cpp)
set(YOLOV8_HEADERS src/YOLOv8/yolov8.h)
add_library(YOLOv8Lib ${YOLOV8_SOURCES} ${YOLOV8_HEADERS})

target_link_libraries(YOLOv8Lib PRIVATE openvino::runtime ${OpenCV_LIBS})
target_include_directories(YOLOv8Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/YOLOv8)

# -------------- POSE --------------
set(EPNP_SOURCES src/EPnP/EPnP.cpp)
set(EPNP_HEADERS src/EPnP/EPnP.h)
add_library(EPnPLib ${EPNP_SOURCES} ${EPNP_HEADERS})

target_link_libraries(EPnPLib PRIVATE ${OpenCV_LIBS})
target_include_directories(EPnPLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/EPnP)

# -------------- EXECUTABLE --------------
add_executable(inference src/inference.cpp)
target_link_libraries(inference PRIVATE YOLOv8Lib openvino::runtime ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

add_executable(slab_score src/slab_score.cpp)
target_link_libraries(slab_score PRIVATE EPnPLib ${OpenCV_LIBS} nlohmann_json::nlohmann_json)