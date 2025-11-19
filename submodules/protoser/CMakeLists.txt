# ProtoSer 子模块 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(protoser)

# 查找源文件
file(GLOB_RECURSE PROTO_SRC_FILES 
  param/*.cc  
)

# 创建共享库
add_library(${PROJECT_NAME} SHARED ${PROTO_SRC_FILES})

# 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 包含目录
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/param
)

# 包含 Protobuf 头文件
target_include_directories(${PROJECT_NAME} PRIVATE
  ${Protobuf_INCLUDE_DIRS}
)

# 链接 Protobuf 库
target_link_libraries(${PROJECT_NAME} 
  ${Protobuf_LIBRARIES}
)

# 设置 rpath
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH_USE_ORIGIN TRUE
)
