# FastDDSSer 子模块 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

project(fastddsser)

# 查找源文件
file(GLOB_RECURSE FASTDDS_SRC_FILES 
  data/*.cxx
)

# 创建共享库
add_library(${PROJECT_NAME} SHARED ${FASTDDS_SRC_FILES})

# 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 包含目录
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/data
)

# 链接外部库（使用相对路径）
target_link_libraries(${PROJECT_NAME} 
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib/libfastcdr.so
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib/libfastrtps.so
  ${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib/libtinyxml2.so
)

# 设置 rpath，确保运行时使用工程目录下的库（使用相对路径）
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib:${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib"
  INSTALL_RPATH "${CMAKE_SOURCE_DIR}/submodules/thirdparty/fastdds/lib:${CMAKE_SOURCE_DIR}/submodules/thirdparty/tinyxml2/lib"
  BUILD_RPATH_USE_ORIGIN TRUE
)

