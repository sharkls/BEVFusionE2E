# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0146 OLD)
project(bevfusion)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

set(spconv_include ../libraries/3DSparseConvolution/libspconv/include)
set(spconv_lib ../libraries/3DSparseConvolution/libspconv/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION})
message(STATUS "spconv_lib = ${spconv_lib}")

if("$ENV{SPCONV_CUDA_VERSION}" EQUAL "11.4")
  # For cuda 11.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")
else()
  # For cuda 12.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
endif()

# Set CUDA SM architecture; default to 86 if env CUDASM is empty
if("$ENV{CUDASM}" STREQUAL "")
  set(CUDA_ARCH 86)
else()
  set(CUDA_ARCH $ENV{CUDASM})
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}
)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  ../dependencies/stb
  ../dependencies/dlpack/include
  ../dependencies/pybind11/include
  ../libraries/cuOSD/src
  ${spconv_include}
  src/common
  src/
  submodules/fastddsser/data
  /share/Code/Lidar_AI_Solution/libraries/fastdds/include
  /share/Code/Lidar_AI_Solution/libraries/tinyxml2/include
)

link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ${spconv_lib}
  build
  $ENV{Python_Lib}
  /share/Code/Lidar_AI_Solution/libraries/fastdds/lib
  /share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib
)

file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/onnx/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libfastddsser.so ##########################
file(GLOB_RECURSE FASTDDS_SRC_FILES 
  submodules/fastddsser/data/*.cxx
)

add_library(fastddsser SHARED ${FASTDDS_SRC_FILES})

# 使用工程目录下的库，而不是系统其他位置的库
target_link_libraries(fastddsser 
  /share/Code/Lidar_AI_Solution/libraries/fastdds/lib/libfastcdr.so
  /share/Code/Lidar_AI_Solution/libraries/fastdds/lib/libfastrtps.so
  /share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib/libtinyxml2.so
)

# 设置 rpath，确保运行时使用工程目录下的库
set_target_properties(fastddsser PROPERTIES
  BUILD_RPATH "/share/Code/Lidar_AI_Solution/libraries/fastdds/lib:/share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib"
  INSTALL_RPATH "/share/Code/Lidar_AI_Solution/libraries/fastdds/lib:/share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib"
)

target_include_directories(fastddsser PUBLIC
  submodules/fastddsser/data
)

####################### libprotoser.so ##########################
file(GLOB_RECURSE PROTO_SRC_FILES 
  submodules/protoser/param/*.cc  
)

add_library(protoser SHARED ${PROTO_SRC_FILES})

# 使用工程目录下的库
target_link_libraries(protoser 
  /share/Code/Lidar_AI_Solution/libraries/fastdds/lib/libfastcdr.so
  /share/Code/Lidar_AI_Solution/libraries/fastdds/lib/libfastrtps.so
  /share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib/libtinyxml2.so
)

# 设置 rpath
set_target_properties(protoser PROPERTIES
  BUILD_RPATH "/share/Code/Lidar_AI_Solution/libraries/fastdds/lib:/share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib"
  INSTALL_RPATH "/share/Code/Lidar_AI_Solution/libraries/fastdds/lib:/share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib"
)

target_include_directories(protoser PUBLIC
  submodules/protoser/param
)

#######################################################################

####################### libbevfusion_alg.so ##########################
cuda_add_library(bevfusion_alg SHARED 
  ${CORE_FILES}
  src/bevfusion/bevfusion.cpp
  src/bevfusion_alg_implement.cpp
  submodules/protoser/param/BEVFusionConfig.pb.cc
  submodules/protoser/param/CameraConfig.pb.cc
)

target_link_libraries(bevfusion_alg
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  fastddsser
)
#######################################################################


####################### test_bevfusion_alg program #############################
cuda_add_executable(test_bevfusion_alg 
  test_bevfusion_alg.cpp
  ${VISUALIZE_FILES}
)

# 设置 rpath，优先使用构建目录和工程目录下的库
# 使用 $ORIGIN 确保相对路径正确，构建目录必须放在最前面
set_target_properties(test_bevfusion_alg PROPERTIES
  BUILD_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}:/share/Code/Lidar_AI_Solution/libraries/fastdds/lib:/share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib"
  INSTALL_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}:/share/Code/Lidar_AI_Solution/libraries/fastdds/lib:/share/Code/Lidar_AI_Solution/libraries/tinyxml2/lib"
  BUILD_RPATH_USE_ORIGIN TRUE
)

target_link_libraries(test_bevfusion_alg
  fastddsser
  bevfusion_alg
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################


########################## custom_layernorm.so ################################
# 为custom_layernorm设置CUDA包含路径
set(CUSTOM_LAYERNORM_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
if(DEFINED ENV{TensorRT_Inc} AND NOT "$ENV{TensorRT_Inc}" STREQUAL "")
  set(CUSTOM_LAYERNORM_NVCC_FLAGS "${CUSTOM_LAYERNORM_NVCC_FLAGS} -I$ENV{TensorRT_Inc}")
endif()
if(DEFINED ENV{CUDA_Inc} AND NOT "$ENV{CUDA_Inc}" STREQUAL "")
  set(CUSTOM_LAYERNORM_NVCC_FLAGS "${CUSTOM_LAYERNORM_NVCC_FLAGS} -I$ENV{CUDA_Inc}")
endif()

cuda_add_library(custom_layernorm SHARED
  src/plugins/custom_layernorm.cu
)

set_target_properties(custom_layernorm PROPERTIES
  CUDA_NVCC_FLAGS "${CUSTOM_LAYERNORM_NVCC_FLAGS}"
)

target_link_libraries(custom_layernorm
  libnvinfer.so
  libnvinfer_plugin.so
)
#######################################################################


########################## libpybev.so ################################
if(CMAKE_BUILD_PYTHON STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )
  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
  )
endif()
#######################################################################