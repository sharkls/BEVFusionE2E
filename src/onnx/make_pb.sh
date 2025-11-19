#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# 获取当前脚本所在目录的绝对路径，并将其赋值给变量project_folder。
project_folder=$(realpath $(dirname ${BASH_SOURCE[-1]}))
cd $project_folder

# 设置protoc编译器的路径，通常用于处理Protocol Buffers文件。
protoc=/usr/bin/protoc
# protoc=/ultralytics/c++/Submodules/TPL/protobuf/bin/protoc
mkdir -p pbout

# 使用protoc编译onnx-ml.proto文件，并将生成的C++代码输出到pbout目录。
$protoc onnx-ml.proto --cpp_out=pbout

# 使用protoc编译onnx-operators-ml.proto文件，并将生成的C++代码输出到pbout目录。
$protoc onnx-operators-ml.proto --cpp_out=pbout

mv pbout/onnx-ml.pb.cc onnx-ml.pb.cpp
mv pbout/onnx-operators-ml.pb.cc onnx-operators-ml.pb.cpp
mv pbout/*.h ./

rm -rf pbout