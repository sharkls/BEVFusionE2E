/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_runtime.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "src/export_bevfusion_alglib.h"
#include "src/common/check.hpp"
#include "src/common/tensor.hpp"
#include "src/common/timer.hpp"
#include "src/common/visualize.hpp"
#include "src/bevfusion/head-transbbox.hpp"

static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                              "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[200];
    sprintf(path, "%s/%s", root.c_str(), file_names[i]);

    int width, height, channels;
    images.push_back(stbi_load(path, &width, &height, &channels, 0));
    // printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
                      cudaStream_t stream) 
{
  std::vector<nv::Prediction> predictions(bboxes.size());
  memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

  int padding = 300;
  int lidar_size = 1024;
  int content_width = lidar_size + padding * 3;
  int content_height = 1080;
  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f;
  bev_artist_param.image_stride = scene_artist_param.stride;

  auto points = lidar_points.to_device();
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
  bev_visualizer->draw_prediction(predictions, false);
  bev_visualizer->draw_ego();
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
  memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
         sizeof(float) * image_artist_param.viewport_nx4x4.size());

  int gap = 0;
  int camera_width = 500;
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
  int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    int ox = offset_cameras[icamera][0] + content_width / 2;
    int oy = offset_cameras[icamera][1] + content_height / 2;
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(icamera, predictions, xflip);

    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                   device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                     device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  printf("Save to %s\n", save_path.c_str());
  stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                 scene_device_image.to_host(stream).ptr(), 100);
}

// 算法结果回调函数
void algorithmCallback(const CAlgResult& result, void* handle) {
    printf("Received algorithm result with %zu frames\n", result.vecFrameResult().size());
    
    for (size_t i = 0; i < result.vecFrameResult().size(); ++i) {
        const auto& frame = result.vecFrameResult()[i];
        printf("Frame %zu: %zu objects detected\n", i, frame.vecObjectResult().size());
        
        for (size_t j = 0; j < frame.vecObjectResult().size(); ++j) {
            const auto& obj = frame.vecObjectResult()[j];
            printf("  Object %zu: %s at (%.2f, %.2f, %.2f) with confidence %.3f\n", 
                   j, obj.strClass().c_str(), obj.x(), obj.y(), obj.z(), obj.confidence());
        }
    }
}

int main(int argc, char** argv) {
    const char* data = "example-data";   // 数据
    const char* model = "resnet50int8";   // 模型
    const char* precision = "int8";        // 精度

    if (argc > 1) data = argv[1];
    if (argc > 2) model = argv[2];
    if (argc > 3) precision = argv[3];

    printf("Testing BEVFusion Algorithm Interface\n");
    printf("Data: %s, Model: %s, Precision: %s\n", data, model, precision);

    // 1. 创建BEVFusion算法对象
    IBEVFusionAlg* bevfusion_alg = CreateBEVFusionAlgObj("");
    if (bevfusion_alg == nullptr) {
        printf("Failed to create BEVFusion algorithm object.\n");
        return -1;
    }

    // 2. 初始化算法
    if (!bevfusion_alg->initAlgorithm("", algorithmCallback, nullptr)) {
        printf("Failed to initialize BEVFusion algorithm.\n");
        delete bevfusion_alg;
        return -1;
    }

    // 3. 参数已通过配置文件自动设置，无需手动调用
    // 注意：setConfidenceThreshold、setTimer、updateCameraParams 方法已被移除
    // 这些参数现在通过 BEVFusionAlgConfig.conf 配置文件自动加载

    // 4. 解析数据路径（处理从build目录运行的情况）
    std::filesystem::path data_path(data);
    std::filesystem::path current_path = std::filesystem::current_path();
    
    // 如果路径不存在，尝试从项目根目录查找
    if (!std::filesystem::exists(data_path)) {
        // 尝试向上查找项目根目录（处理在build目录下运行的情况）
        std::filesystem::path test_path = current_path.parent_path() / data;
        if (std::filesystem::exists(test_path)) {
            data_path = test_path;
            printf("Found data directory at: %s\n", data_path.string().c_str());
        } else {
            // 如果还是找不到，尝试当前目录的父目录
            test_path = current_path / data;
            if (std::filesystem::exists(test_path)) {
                data_path = test_path;
                printf("Found data directory at: %s\n", data_path.string().c_str());
            } else {
                printf("Error: Cannot find data directory: %s\n", data);
                printf("Tried paths:\n");
                printf("  1. %s\n", std::filesystem::path(data).string().c_str());
                printf("  2. %s\n", (current_path.parent_path() / data).string().c_str());
                printf("  3. %s\n", (current_path / data).string().c_str());
                delete bevfusion_alg;
                return -1;
            }
        }
    }
    
    std::string data_dir = data_path.string();

    // 6. 加载图像和激光雷达数据
    auto images = load_images(data_dir);
    std::string points_path = nv::format("%s/points.tensor", data_dir.c_str());
    printf("Loading lidar points from: %s\n", points_path.c_str());
    
    auto lidar_points = nv::Tensor::load(points_path, false);
    
    // 检查 tensor 是否成功加载
    if (lidar_points.empty()) {
        printf("Failed to load lidar points tensor from %s\n", points_path.c_str());
        printf("Tensor is empty. Please check if the file exists and is valid.\n");
        delete bevfusion_alg;
        return -1;
    }
    
    printf("Loaded lidar points: shape=[%ld, %ld], dtype=%d, device=%d, num_points=%ld\n", 
           lidar_points.size(0), lidar_points.size(1), 
           static_cast<int>(lidar_points.dtype()), 
           lidar_points.device() ? 1 : 0,
           lidar_points.size(0));

    // 7. 准备输入数据结构
    BEVFusionInputData input_data;
    input_data.images = images;
    
    // 检查数据类型并获取指针
    // forward 函数期望主机上的 half* 指针
    if (lidar_points.dtype() == nv::DataType::Float16) {
        // 如果 tensor 在设备上，需要先复制到主机
        if (lidar_points.device()) {
            printf("Warning: lidar_points is on device, copying to host...\n");
            auto lidar_points_host = lidar_points.to_host();
            if (lidar_points_host.empty()) {
                printf("Failed to copy lidar_points to host.\n");
                delete bevfusion_alg;
                return -1;
            }
            input_data.lidar_points = lidar_points_host.ptr<nvtype::half>();
        } else {
            input_data.lidar_points = lidar_points.ptr<nvtype::half>();
        }
    } else {
        printf("Error: lidar_points dtype is %d, expected Float16 (%d)\n", 
               static_cast<int>(lidar_points.dtype()), 
               static_cast<int>(nv::DataType::Float16));
        printf("Cannot proceed with incompatible data type.\n");
        delete bevfusion_alg;
        return -1;
    }
    
    if (input_data.lidar_points == nullptr) {
        printf("Error: Failed to get lidar_points pointer.\n");
        delete bevfusion_alg;
        return -1;
    }
    
    input_data.num_points = lidar_points.size(0);

    // 8. 运行算法（包含预热和多次执行以测试性能）
    printf("Running BEVFusion algorithm...\n");
    
    // 8.1 预热推理（第一次执行，不计算时间，用于初始化 CUDA kernels 和缓存）
    printf("Warmup inference...\n");
    bevfusion_alg->runAlgorithm(&input_data);
    
    // 8.2 多次执行以评估推理时间（类似 CUDA-BEVFusion 的做法）
    printf("Evaluating inference time (5 runs)...\n");
    for (int i = 0; i < 5; ++i) {
        bevfusion_alg->runAlgorithm(&input_data);
    }

    // 9. 清理资源
    free_images(images);
    delete bevfusion_alg;

    printf("BEVFusion algorithm test completed successfully.\n");
    return 0;
}