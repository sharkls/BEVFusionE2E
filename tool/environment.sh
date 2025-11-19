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

# export CUDA_VISIBLE_DEVICES=2

export TensorRT_Lib=/mnt/env/TensorRT-10.2.0.19/lib
export TensorRT_Inc=/mnt/env/TensorRT-10.2.0.19/include
export TensorRT_Bin=/mnt/env/TensorRT-10.2.0.19/bin

export CUDA_Lib=/usr/local/cuda-12.4/lib64
export CUDA_Inc=/usr/local/cuda-12.4/include
export CUDA_Bin=/usr/local/cuda-12.4/bin
export CUDA_HOME=/usr/local/cuda-12.4

export CUDNN_Lib=/usr/local/cuda-12.4/lib64

# export TensorRT_Lib=${TENSORRT_LIB}
# export TensorRT_Inc=${TENSORRT_INCLUDE}
# export TensorRT_Bin=${TENSORRT_BIN}

# export CUDA_Lib=${CUDA_LIB}
# export CUDA_Inc=${CUDA_INCLUDE}
# export CUDA_Bin=${CUDA_BIN}

# export CUDNN_Lib=${CUDNN_LIB}

# Just used to distinguish the libspconv version, it doesn't affect the version of cuda used by your application
# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.6
export SPCONV_CUDA_VERSION=12.6

# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet50int8

# fp16/int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=example-data
export USE_Python=OFF

# check the configuration path
# clean the configuration status
export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "||  CUDNN: $CUDNN_Lib"
echo "=========================================================="

BuildDirectory=`pwd`/build

if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`
    echo Find Python_Inc: $Python_Inc
    echo Find Python_Lib: $Python_Lib
    echo Find Python_Soname: $Python_Soname
fi

export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
# 优先使用工程目录下的库，确保链接到正确的 fastdds 和 tinyxml2 库
# 构建目录必须放在最前面，确保优先使用构建目录中的 libfastddsser.so
# 过滤掉可能存在的 FastDDS-Demo 路径，避免链接到错误的库
# 使用相对路径，基于脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRDPARTY_FASTDDS_LIB="$PROJECT_ROOT/submodules/thirdparty/fastdds/lib"
THIRDPARTY_TINYXML2_LIB="$PROJECT_ROOT/submodules/thirdparty/tinyxml2/lib"
CLEANED_LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "FastDDS-Demo" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$BuildDirectory:$THIRDPARTY_FASTDDS_LIB:$THIRDPARTY_TINYXML2_LIB:$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:${CLEANED_LD_LIBRARY_PATH}
export PYTHONPATH=$BuildDirectory:$PYTHONPATH
export ConfigurationStatus=Success

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    echo "Current CUDA SM: $cudasm"
fi

export CUDASM=$cudasm

echo Configuration done!