# BEVFusion 核心库 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(bevfusion)

# 查找 BEVFusion 核心文件
file(GLOB_RECURSE BEVFUSION_CORE_FILES 
  bevfusion/*.cu 
  bevfusion/*.cpp
  onnx/*.cpp
  common/tensor.cu
  common/tensorrt.cpp
)

# 查找算法接口实现文件
file(GLOB_RECURSE BEVFUSION_ALG_FILES
  bevfusion_alg_implement.cpp
  export_bevfusion_alglib.cpp
)

# 查找 Protobuf 生成的文件
file(GLOB_RECURSE PROTO_FILES
  ${CMAKE_SOURCE_DIR}/submodules/protoser/param/*.cc
)

# 创建共享库
cuda_add_library(${PROJECT_NAME} SHARED 
  ${BEVFUSION_CORE_FILES}
  ${BEVFUSION_ALG_FILES}
  ${PROTO_FILES}
)

# 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  OUTPUT_NAME "bevfusion"
  VERSION 1.0.0
  SOVERSION 1
)

# 包含目录（对外暴露的接口）
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/submodules/fastddsser/data>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/submodules/protoser/param>
  $<INSTALL_INTERFACE:include>
)

# 内部使用的包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  ${CMAKE_SOURCE_DIR}/../dependencies/stb
  ${CMAKE_SOURCE_DIR}/../dependencies/dlpack/include
  ${spconv_include}
  ${CMAKE_SOURCE_DIR}/../libraries/cuOSD/src
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/include
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/include
)

# 链接库
target_link_libraries(${PROJECT_NAME}
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  fastddsser
  protoser
)

# 设置 rpath，确保运行时能找到依赖库（使用相对路径）
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib:${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib"
  INSTALL_RPATH "$ORIGIN:${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib:${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib"
  BUILD_RPATH_USE_ORIGIN TRUE
)

# 安装规则
install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES 
  export_bevfusion_alglib.h
  DESTINATION include
)

install(DIRECTORY 
  common/
  DESTINATION include/common
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY 
  ${CMAKE_SOURCE_DIR}/submodules/fastddsser/data/
  DESTINATION include
  FILES_MATCHING PATTERN "CAlgResult.h" PATTERN "CDataBase.h"
)

#######################################################################
# 自定义插件（与 BEVFusion 内部实现相关）
#######################################################################

# custom_layernorm 插件
set(CUSTOM_LAYERNORM_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
if(DEFINED ENV{TensorRT_Inc} AND NOT "$ENV{TensorRT_Inc}" STREQUAL "")
  set(CUSTOM_LAYERNORM_NVCC_FLAGS "${CUSTOM_LAYERNORM_NVCC_FLAGS} -I$ENV{TensorRT_Inc}")
endif()
if(DEFINED ENV{CUDA_Inc} AND NOT "$ENV{CUDA_Inc}" STREQUAL "")
  set(CUSTOM_LAYERNORM_NVCC_FLAGS "${CUSTOM_LAYERNORM_NVCC_FLAGS} -I$ENV{CUDA_Inc}")
endif()

cuda_add_library(custom_layernorm SHARED
  plugins/custom_layernorm.cu
)

set_target_properties(custom_layernorm PROPERTIES
  CUDA_NVCC_FLAGS "${CUSTOM_LAYERNORM_NVCC_FLAGS}"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(custom_layernorm
  libnvinfer.so
  libnvinfer_plugin.so
)

#######################################################################
# Python 绑定（可选，属于 BEVFusion 内部实现）
#######################################################################

if(CMAKE_BUILD_PYTHON STREQUAL "ON")
  cuda_add_library(pybev SHARED
    python.cpp
  )
  
  set_target_properties(pybev PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  
  target_link_libraries(pybev
    bevfusion
    $ENV{Python_Soname}
  )
endif()


