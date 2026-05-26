cmake_minimum_required(VERSION 3.12)
project(b2_controller)

# 添加 Unitree SDK 和系统库的搜索路径
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake" "/usr")

# 强制 spdlog 使用外部 fmt 库
add_definitions(-DSPDLOG_FMT_EXTERNAL=1)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(unitree_sdk2 REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
endif()

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
target_include_directories(${PROJECT_NAME}_lib
  PUBLIC
    /usr/include/eigen3
    include
    ${ONNXRUNTIME_DIR}/include
    ${PROJECT_SOURCE_DIR}/../../include/
)
target_link_libraries(${PROJECT_NAME}_lib
  PUBLIC
    unitree_sdk2
    rt
    pthread
    boost_program_options
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

add_executable(b2_ctrl main.cpp)
target_link_libraries(b2_ctrl PRIVATE ${PROJECT_NAME}_lib)
set_target_properties(b2_ctrl PROPERTIES
  BUILD_RPATH "/opt/unitree_robotics/lib;${ONNXRUNTIME_DIR}/lib"
  INSTALL_RPATH "/opt/unitree_robotics/lib;${ONNXRUNTIME_DIR}/lib"
)