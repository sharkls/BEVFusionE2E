/*******************************************************
 文件名：bevfusion_alg_implement.h
 作者：sharkls
 描述：BEVFusion算法接口实现类的头文件
 版本：v2.0
 日期：2025-01-15
 *******************************************************/
#pragma once

#include "export_bevfusion_alglib.h"
#include "bevfusion/bevfusion.hpp"
#include <memory>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>

// 配置参数结构体 - 从protobuf配置文件加载
struct BEVFusionConfig {
    float confidence_threshold = 0.12f;
    bool enable_timer = true;
    std::string model_name = "resnet50int8";
    std::string precision = "int8";
    
    // 相机配置
    uint32_t num_camera = 6;
    std::vector<std::vector<float>> camera_intrinsics;      // 每个相机的内参矩阵
    std::vector<std::vector<float>> camera2lidar;           // 每个相机的变换矩阵
    std::vector<std::vector<float>> lidar2image;            // 每个相机的变换矩阵
    std::vector<std::vector<float>> img_aug_matrix;         // 每个相机的增强矩阵
    std::vector<uint32_t> image_widths, image_heights;        // 每个相机的图像尺寸
    std::vector<uint32_t> output_widths, output_heights;      // 每个相机的输出尺寸
    std::vector<float> resize_lims;                         // 每个相机的缩放限制
    std::vector<std::vector<float>> means, stds;            // 每个相机的归一化参数
    std::vector<float> scale_factors, offsets;              // 每个相机的缩放因子和偏移
    std::string interpolation = "Bilinear";                // 图像插值方法，"Bilinear" 或 "Nearest"
    
    // 激光雷达配置
    std::vector<float> lidar_min_range = {-54.0f, -54.0f, -5.0f};
    std::vector<float> lidar_max_range = {54.0f, 54.0f, 3.0f};
    std::vector<float> lidar_voxel_size = {0.075f, 0.075f, 0.2f};
    std::vector<float> lidar_grid_size = {1440, 1440, 40};
    uint32_t max_points_per_voxel = 10;
    uint32_t max_points = 300000;
    uint32_t max_voxels = 160000;
    uint32_t num_feature = 5;
    std::string lidar_model_path = "model/resnet50int8/lidar.backbone.xyz.onnx";
    std::string coordinate_order = "XYZ";
    
    // 几何配置
    std::vector<float> geometry_xbound = {-54.0f, 54.0f, 0.3f};
    std::vector<float> geometry_ybound = {-54.0f, 54.0f, 0.3f};
    std::vector<float> geometry_zbound = {-10.0f, 10.0f, 20.0f};
    std::vector<float> geometry_dbound = {1.0f, 60.0f, 0.5f};
    uint32_t geometry_image_width = 704;
    uint32_t geometry_image_height = 256;
    uint32_t geometry_feat_width = 88;
    uint32_t geometry_feat_height = 32;
    uint32_t geometry_num_camera = 6;
    std::vector<uint32_t> geometry_dim = {360, 360, 80};
    
    // 后处理配置
    uint32_t out_size_factor = 8;
    std::vector<float> pc_range = {-54.0f, -54.0f};
    std::vector<float> post_center_range_start = {-61.2f, -61.2f, -10.0f};
    std::vector<float> post_center_range_end = {61.2f, 61.2f, 10.0f};
    std::vector<float> postprocessor_voxel_size = {0.075f, 0.075f};
    bool sorted_bboxes = true;
    
    // 模型路径配置
    std::string camera_backbone_path = "model/resnet50int8/build/camera.backbone.plan";
    std::string lidar_backbone_path = "model/resnet50int8/lidar.backbone.xyz.onnx";
    std::string fuser_path = "model/resnet50int8/build/fuser.plan";
    std::string head_bbox_path = "model/resnet50int8/build/head.bbox.plan";
    std::string camera_vtransform_path = "model/resnet50int8/build/camera.vtransform.plan";
};

// BEVFusion算法接口实现类
class BEVFusionAlgImplement : public IBEVFusionAlg {
public:
    BEVFusionAlgImplement();
    virtual ~BEVFusionAlgImplement();

    // 初始化算法接口对象
    virtual bool initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd) override;

    // 执行算法函数
    virtual void runAlgorithm(void* p_pSrcData) override;

private:
    std::shared_ptr<bevfusion::Core> core_;
    cudaStream_t stream_;
    AlgCallback alg_callback_;
    void* handle_;
    
    // 配置文件解析方法 - 使用protobuf文本格式加载所有参数
    bool loadConfigFromFile(const std::string& exe_path, BEVFusionConfig& config, std::filesystem::path& project_root);
    
    // 创建BEVFusion核心的辅助函数 - 使用配置文件参数
    std::shared_ptr<bevfusion::Core> create_core_with_config(const BEVFusionConfig& config);
    
    // 将检测结果转换为CAlgResult格式
    void convertBBoxesToResult(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, CAlgResult& result);
};

