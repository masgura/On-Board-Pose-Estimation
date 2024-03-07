cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(yolov8_libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# -------------- OpenCV --------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- libtorch --------------
set(CMAKE_PREFIX_PATH /home/mauro/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${TORCH_INCLUDE_DIRS})

add_executable(obj_det "${CMAKE_CURRENT_SOURCE_DIR}/obj_det.cc")
target_link_libraries(obj_det ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET obj_det PROPERTY CXX_STANDARD 17)

add_executable(main "${CMAKE_CURRENT_SOURCE_DIR}/main.cc")
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)

add_executable(EPnP "${CMAKE_CURRENT_SOURCE_DIR}/EPnP.cc")
target_link_libraries(EPnP ${OpenCV_LIBS})
set_property(TARGET EPnP PROPERTY CXX_STANDARD 17)
