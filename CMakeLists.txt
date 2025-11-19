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

# 顶层项目配置
project(bevfusion_top)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

# 查找依赖包
find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

# spconv 配置（使用相对路径）
set(spconv_include ${CMAKE_SOURCE_DIR}/submodules/thirdparty/3DSparseConvolution/libspconv/include)
set(spconv_lib ${CMAKE_SOURCE_DIR}/submodules/thirdparty/3DSparseConvolution/libspconv/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION})
message(STATUS "spconv_lib = ${spconv_lib}")

# 编译选项
if("$ENV{SPCONV_CUDA_VERSION}" EQUAL "11.4")
  # For cuda 11.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")
else()
  # For cuda 12.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
endif()

# CUDA 架构设置
if("$ENV{CUDASM}" STREQUAL "")
  set(CUDA_ARCH 86)
else()
  set(CUDA_ARCH $ENV{CUDASM})
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}
)

# 全局包含目录
include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  ${CMAKE_SOURCE_DIR}/../dependencies/stb
  ${CMAKE_SOURCE_DIR}/../dependencies/dlpack/include
  ${CMAKE_SOURCE_DIR}/../dependencies/pybind11/include
  ${CMAKE_SOURCE_DIR}/../libraries/cuOSD/src
  ${spconv_include}
  ${CMAKE_SOURCE_DIR}/src/common
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/submodules/fastddsser/data
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/include
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/include
)

# 全局链接目录
link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ${spconv_lib}
  ${CMAKE_BINARY_DIR}
  $ENV{Python_Lib}
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib
)

# 将 spconv_include 传递给子项目
set(spconv_include ${spconv_include} CACHE INTERNAL "spconv include directory")

#######################################################################
# 添加子模块
#######################################################################

# 添加 fastddsser 子模块
add_subdirectory(submodules/fastddsser)

# 添加 protoser 子模块
add_subdirectory(submodules/protoser)

# 添加 bevfusion 核心库（src 目录下的项目）
add_subdirectory(src)

#######################################################################
# 测试程序
#######################################################################

# 可视化相关文件
file(GLOB_RECURSE VISUALIZE_FILES 
  ${CMAKE_SOURCE_DIR}/../libraries/cuOSD/src/*.cpp
  ${CMAKE_SOURCE_DIR}/../libraries/cuOSD/src/*.cu
  ${CMAKE_SOURCE_DIR}/src/common/visualize.cu
)

# 创建测试可执行文件
cuda_add_executable(test_bevfusion_alg 
  test_bevfusion_alg.cpp
  ${VISUALIZE_FILES}
)

# 设置 rpath，确保运行时能找到所有依赖库（使用相对路径）
set_target_properties(test_bevfusion_alg PROPERTIES
  BUILD_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib:${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib"
  INSTALL_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib:${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib"
  BUILD_RPATH_USE_ORIGIN TRUE
)

# 链接库（只需要链接 bevfusion，它会自动传递依赖）
target_link_libraries(test_bevfusion_alg
  bevfusion
)