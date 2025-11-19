#!/bin/bash
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

# configure the environment
. tool/environment.sh

# 检查配置状态，如果不是成功，则输出错误信息并退出。
if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# tensorrt version 获取TensorRT的版本号
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

# resnet50/resnet50-int8/swint-tiny # 设置模型的基本路径，使用调试模型的名称。
base=model/$DEBUG_MODEL

# fp16/int8 设置精度，使用调试精度的值。
precision=$DEBUG_PRECISION

# precision flags 根据精度设置TensorRT执行的标志。如果精度是int8，则添加相应的标志。
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

# 获取ONNX模型的输入和输出数量。
function get_onnx_number_io(){

    # $1=model
    model=$1

    # 检查模型文件是否存在，如果不存在则输出错误信息并返回。
    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    # 使用Python脚本加载ONNX模型并获取输入和输出的数量。
    number_of_input=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

# 编译TensorRT模型，接收模型名称、精度标志、输入输出数量和额外标志。
function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    # $5: extra_flags
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    extra_flags=$5
    result_save_directory=$base/build
    onnx=$base/$name.onnx

    # 检查模型的计划文件是否已存在，如果存在则输出信息并返回。
    if [ -f "${result_save_directory}/$name.plan" ]; then
        echo Model ${result_save_directory}/$name.plan already build 🙋🙋🙋.
        return
    fi
    
    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    # echo $number_of_input  $number_of_output

    # 根据输入数量构建输入格式标志。
    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    # 根据输出数量构建输出格式标志。
    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    # 去掉最后一个逗号。
    input_flags=${input_flags%?}
    output_flags=${output_flags%?}

    # 构建TensorRT编译命令。
    cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} ${extra_flags} \
        --saveEngine=${result_save_directory}/$name.plan \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"

    # 创建保存结果的目录，并输出正在构建模型的信息。
    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    
    # 执行编译命令，并将输出重定向到日志文件。如果编译失败，则输出错误信息并退出。
    trtexec $cmd > ${result_save_directory}/$name.log 2>&1
    if [ $? != 0 ]; then
        echo 😥 Failed to build model ${result_save_directory}/$name.plan.
        echo You can check the error message by ${result_save_directory}/$name.log 
        exit 1
    fi
}

# maybe int8 / fp16     编译两个模型，使用动态精度标志。
compile_trt_model "camera.backbone" "$trtexec_dynamic_flags" 2 2
compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1

# fp16 only 编译一个仅使用fp16精度的模型。
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1

# 编译一个特定的模型，可能会遇到TensorRT的bug，但速度更快。
# for myelin layernorm head.bbox, may occur a tensorrt bug at layernorm fusion but faster
compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6

# 编译另一个版本的模型，准确但速度较慢。
# for layernorm version head.bbox.onnx, accurate but slower
# compile_trt_model "head.bbox.layernormplugin" "$trtexec_fp16_flags" 1 6 "--plugins=libcustom_layernorm.so"
