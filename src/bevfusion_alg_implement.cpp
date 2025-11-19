/*******************************************************
 文件名：bevfusion_alg_implement.cpp
 作者：sharkls
 描述：BEVFusion算法接口实现类的实现文件
 版本：v2.0
 日期：2025-01-15
 *******************************************************/
#include "bevfusion_alg_implement.h"
#include "common/check.hpp"
#include "common/timer.hpp"
#include "common/tensor.hpp"
#include <fstream>
#include <filesystem>
#include <google/protobuf/text_format.h>
#include "../submodules/protoser/param/BEVFusionConfig.pb.h"
#include <dlfcn.h>

// 构造函数
BEVFusionAlgImplement::BEVFusionAlgImplement() 
    : core_(nullptr), stream_(nullptr), alg_callback_(nullptr), handle_(nullptr) {
}

// 析构函数
BEVFusionAlgImplement::~BEVFusionAlgImplement() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

// 初始化算法接口对象
bool BEVFusionAlgImplement::initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd) {
    alg_callback_ = alg_cb;
    handle_ = hd;
    
    // 加载配置文件
    BEVFusionConfig config;
    std::filesystem::path project_root;  // 保存项目根目录路径
    if (!loadConfigFromFile(exe_path, config, project_root)) {
        printf("Warning: Failed to load config file, using default values.\n");
    }
    
    // 将模型路径转换为绝对路径（基于项目根目录）
    if (!project_root.empty()) {
        if (!config.camera_backbone_path.empty() && !std::filesystem::path(config.camera_backbone_path).is_absolute()) {
            config.camera_backbone_path = (project_root / config.camera_backbone_path).string();
        }
        if (!config.lidar_backbone_path.empty() && !std::filesystem::path(config.lidar_backbone_path).is_absolute()) {
            config.lidar_backbone_path = (project_root / config.lidar_backbone_path).string();
        }
        if (!config.lidar_model_path.empty() && !std::filesystem::path(config.lidar_model_path).is_absolute()) {
            config.lidar_model_path = (project_root / config.lidar_model_path).string();
        }
        if (!config.fuser_path.empty() && !std::filesystem::path(config.fuser_path).is_absolute()) {
            config.fuser_path = (project_root / config.fuser_path).string();
        }
        if (!config.head_bbox_path.empty() && !std::filesystem::path(config.head_bbox_path).is_absolute()) {
            config.head_bbox_path = (project_root / config.head_bbox_path).string();
        }
        if (!config.camera_vtransform_path.empty() && !std::filesystem::path(config.camera_vtransform_path).is_absolute()) {
            config.camera_vtransform_path = (project_root / config.camera_vtransform_path).string();
        }
    }
    
    // 加载自定义层库
    dlopen("libcustom_layernorm.so", RTLD_NOW);
    
    // 创建CUDA流
    cudaStreamCreate(&stream_);
    
    // 创建BEVFusion核心 - 使用配置文件中的参数
    core_ = create_core_with_config(config);
    if (core_ == nullptr) {
        printf("Failed to create BEVFusion core.\n");
        return false;
    }
    
    // 设置计时器
    core_->set_timer(config.enable_timer);
    
    // 更新相机参数 - 需要将所有相机的矩阵数据合并成连续数组
    // update 函数期望的数据格式：所有相机的矩阵按顺序排列（6个相机 x 4x4矩阵 = 96个float）
    if (!config.camera2lidar.empty() && !config.camera_intrinsics.empty() && 
        !config.lidar2image.empty() && !config.img_aug_matrix.empty() &&
        config.camera2lidar.size() >= config.num_camera) {
        
        // 准备所有相机的矩阵数据（连续内存布局，每个矩阵16个float）
        std::vector<float> all_camera2lidar, all_camera_intrinsics, all_lidar2image, all_img_aug_matrix;
        all_camera2lidar.reserve(config.num_camera * 16);
        all_camera_intrinsics.reserve(config.num_camera * 16);
        all_lidar2image.reserve(config.num_camera * 16);
        all_img_aug_matrix.reserve(config.num_camera * 16);
        
        for (uint32_t i = 0; i < config.num_camera; i++) {
            if (i < config.camera2lidar.size() && config.camera2lidar[i].size() >= 16) {
                all_camera2lidar.insert(all_camera2lidar.end(), 
                                       config.camera2lidar[i].begin(), 
                                       config.camera2lidar[i].begin() + 16);
            } else {
                printf("Warning: Missing camera2lidar data for camera %d\n", i);
                all_camera2lidar.insert(all_camera2lidar.end(), 16, 0.0f);
            }
            
            if (i < config.camera_intrinsics.size() && config.camera_intrinsics[i].size() >= 16) {
                all_camera_intrinsics.insert(all_camera_intrinsics.end(), 
                                            config.camera_intrinsics[i].begin(), 
                                            config.camera_intrinsics[i].begin() + 16);
            } else {
                printf("Warning: Missing camera_intrinsics data for camera %d\n", i);
                all_camera_intrinsics.insert(all_camera_intrinsics.end(), 16, 0.0f);
            }
            
            if (i < config.lidar2image.size() && config.lidar2image[i].size() >= 16) {
                all_lidar2image.insert(all_lidar2image.end(), 
                                      config.lidar2image[i].begin(), 
                                      config.lidar2image[i].begin() + 16);
            } else {
                printf("Warning: Missing lidar2image data for camera %d\n", i);
                all_lidar2image.insert(all_lidar2image.end(), 16, 0.0f);
            }
            
            if (i < config.img_aug_matrix.size() && config.img_aug_matrix[i].size() >= 16) {
                all_img_aug_matrix.insert(all_img_aug_matrix.end(), 
                                         config.img_aug_matrix[i].begin(), 
                                         config.img_aug_matrix[i].begin() + 16);
            } else {
                printf("Warning: Missing img_aug_matrix data for camera %d\n", i);
                all_img_aug_matrix.insert(all_img_aug_matrix.end(), 16, 0.0f);
            }
        }
        
        // 调用 update 函数，传递所有相机的矩阵数据
        core_->update(all_camera2lidar.data(), all_camera_intrinsics.data(), 
                     all_lidar2image.data(), all_img_aug_matrix.data(), stream_);
    } else {
        printf("Warning: Camera matrix data incomplete, skipping update.\n");
    }
    
    printf("BEVFusion algorithm initialized successfully with confidence_threshold=%.3f, timer=%s.\n", 
           config.confidence_threshold, config.enable_timer ? "enabled" : "disabled");
    return true;
}

// 执行算法函数
void BEVFusionAlgImplement::runAlgorithm(void* p_pSrcData) {
    if (!core_ || !stream_) {
        printf("BEVFusion core not initialized.\n");
        return;
    }
    
    // 解析输入数据
    BEVFusionInputData* input_data = static_cast<BEVFusionInputData*>(p_pSrcData);
    if (input_data == nullptr) {
        printf("Invalid input data pointer.\n");
        return;
    }
    
    // 执行BEVFusion推理
    std::vector<bevfusion::head::transbbox::BoundingBox> bboxes = core_->forward(
        (const unsigned char**)input_data->images.data(), 
        input_data->lidar_points, 
        input_data->num_points, 
        stream_
    );
    
    // 转换结果为CAlgResult格式
    CAlgResult result;
    convertBBoxesToResult(bboxes, result);
    
    // 调用回调函数返回结果
    if (alg_callback_) {
        alg_callback_(result, handle_);
    }
}

// 将检测结果转换为CAlgResult格式
void BEVFusionAlgImplement::convertBBoxesToResult(
    const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, 
    CAlgResult& result) {
    
    CFrameResult frame_result;
    
    for (const auto& bbox : bboxes) {
        CObjectResult obj_result;
        obj_result.strClass("");
        obj_result.x(bbox.position.x);
        obj_result.y(bbox.position.y);
        obj_result.z(bbox.position.z);
        obj_result.w(bbox.size.w);
        obj_result.l(bbox.size.l);
        obj_result.h(bbox.size.h);
        obj_result.yaw(bbox.z_rotation);
        obj_result.confidence(bbox.score);
        obj_result.label(static_cast<uint8_t>(bbox.id));
        
        frame_result.vecObjectResult().push_back(obj_result);
    }
    
    result.vecFrameResult().push_back(frame_result);
}

// 配置文件解析方法 - 使用protobuf文本格式加载所有参数
bool BEVFusionAlgImplement::loadConfigFromFile(const std::string& exe_path, BEVFusionConfig& config, std::filesystem::path& project_root) {
    try {
        // 构建配置文件路径
        std::filesystem::path config_path;
        if (exe_path.empty()) {
            // 如果exe_path为空，尝试从当前工作目录向上查找项目根目录
            std::filesystem::path current_path = std::filesystem::current_path();
            std::filesystem::path test_path = current_path / "configs" / "algorithm" / "BEVFusionAlgConfig.conf";
            
            // 如果当前目录下找不到，尝试向上查找（处理在build目录下运行的情况）
            if (!std::filesystem::exists(test_path)) {
                test_path = current_path.parent_path() / "configs" / "algorithm" / "BEVFusionAlgConfig.conf";
                if (std::filesystem::exists(test_path)) {
                    project_root = current_path.parent_path();
                }
            } else {
                project_root = current_path;
            }
            config_path = test_path;
        } else {
            // 使用exe_path的父目录
            std::filesystem::path exe_file_path(exe_path);
            std::filesystem::path test_path = exe_file_path.parent_path() / "configs" / "algorithm" / "BEVFusionAlgConfig.conf";
            
            // 如果可执行文件目录下找不到，尝试向上查找（处理在build目录下运行的情况）
            if (!std::filesystem::exists(test_path)) {
                test_path = exe_file_path.parent_path().parent_path() / "configs" / "algorithm" / "BEVFusionAlgConfig.conf";
                if (std::filesystem::exists(test_path)) {
                    project_root = exe_file_path.parent_path().parent_path();
                }
            } else {
                project_root = exe_file_path.parent_path();
            }
            config_path = test_path;
        }
        
        printf("Loading config from: %s\n", config_path.string().c_str());
        
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            printf("Config file not found: %s\n", config_path.string().c_str());
            return false;
        }
        
        // 读取整个文件内容
        std::string content((std::istreambuf_iterator<char>(config_file)), 
                           std::istreambuf_iterator<char>());
        config_file.close();
        
        // 使用protobuf文本格式解析
        bevfusion::BEVFusionConfig proto_config;
        if (!google::protobuf::TextFormat::ParseFromString(content, &proto_config)) {
            printf("Failed to parse protobuf config file.\n");
            return false;
        }
        
        // 基础配置参数
        config.confidence_threshold = proto_config.confidence_threshold();
        config.enable_timer = proto_config.enable_timer();
        config.num_camera = proto_config.num_camera();
        config.model_name = proto_config.model_name();
        config.precision = proto_config.precision();
        
        // 相机配置
        config.camera_intrinsics.clear();
        config.camera2lidar.clear();
        config.lidar2image.clear();
        config.img_aug_matrix.clear();
        config.image_widths.clear();
        config.image_heights.clear();
        config.output_widths.clear();
        config.output_heights.clear();
        config.resize_lims.clear();
        config.means.clear();
        config.stds.clear();
        config.scale_factors.clear();
        config.offsets.clear();
        
        for (int i = 0; i < proto_config.cameras_size(); i++) {
            const auto& camera = proto_config.cameras(i);
            
            // 相机内参矩阵
            std::vector<float> intrinsics;
            for (int j = 0; j < camera.intrinsics_size(); j++) {
                intrinsics.push_back(camera.intrinsics(j));
            }
            config.camera_intrinsics.push_back(intrinsics);
            
            // 相机到激光雷达变换矩阵
            std::vector<float> camera2lidar;
            for (int j = 0; j < camera.camera2lidar_size(); j++) {
                camera2lidar.push_back(camera.camera2lidar(j));
            }
            config.camera2lidar.push_back(camera2lidar);
            
            // 激光雷达到图像变换矩阵
            std::vector<float> lidar2image;
            for (int j = 0; j < camera.lidar2image_size(); j++) {
                lidar2image.push_back(camera.lidar2image(j));
            }
            config.lidar2image.push_back(lidar2image);
            
            // 图像增强矩阵
            std::vector<float> img_aug_matrix;
            for (int j = 0; j < camera.img_aug_matrix_size(); j++) {
                img_aug_matrix.push_back(camera.img_aug_matrix(j));
            }
            config.img_aug_matrix.push_back(img_aug_matrix);
            
            // 相机尺寸参数
            config.image_widths.push_back(camera.image_width());
            config.image_heights.push_back(camera.image_height());
            config.output_widths.push_back(camera.output_width());
            config.output_heights.push_back(camera.output_height());
            config.resize_lims.push_back(camera.resize_lim());
            
            // 归一化参数
            std::vector<float> mean, std_val;
            for (int j = 0; j < camera.mean_size(); j++) {
                mean.push_back(camera.mean(j));
            }
            for (int j = 0; j < camera.std_size(); j++) {
                std_val.push_back(camera.std(j));
            }
            config.means.push_back(mean);
            config.stds.push_back(std_val);
            config.scale_factors.push_back(camera.scale_factor());
            config.offsets.push_back(camera.offset());
            
            // 读取插值方法（如果配置了，使用第一个相机的值，因为所有相机应该使用相同的插值方法）
            if (i == 0 && !camera.interpolation().empty()) {
                config.interpolation = camera.interpolation();
            }
        }
        
        // 激光雷达配置
        if (proto_config.has_lidar_config()) {
            const auto& lidar = proto_config.lidar_config();
            
            config.lidar_min_range.clear();
            for (int i = 0; i < lidar.min_range_size(); i++) {
                config.lidar_min_range.push_back(lidar.min_range(i));
            }
            
            config.lidar_max_range.clear();
            for (int i = 0; i < lidar.max_range_size(); i++) {
                config.lidar_max_range.push_back(lidar.max_range(i));
            }
            
            config.lidar_voxel_size.clear();
            for (int i = 0; i < lidar.voxel_size_size(); i++) {
                config.lidar_voxel_size.push_back(lidar.voxel_size(i));
            }
            
            config.lidar_grid_size.clear();
            for (int i = 0; i < lidar.grid_size_size(); i++) {
                config.lidar_grid_size.push_back(lidar.grid_size(i));
            }
            
            config.max_points_per_voxel = lidar.max_points_per_voxel();
            config.max_points = lidar.max_points();
            config.max_voxels = lidar.max_voxels();
            config.num_feature = lidar.num_feature();
            config.lidar_model_path = lidar.lidar_model_path();
            config.coordinate_order = lidar.coordinate_order();
        }
        
        // 几何配置
        if (proto_config.has_geometry_config()) {
            const auto& geometry = proto_config.geometry_config();
            
            config.geometry_xbound.clear();
            for (int i = 0; i < geometry.xbound_size(); i++) {
                config.geometry_xbound.push_back(geometry.xbound(i));
            }
            
            config.geometry_ybound.clear();
            for (int i = 0; i < geometry.ybound_size(); i++) {
                config.geometry_ybound.push_back(geometry.ybound(i));
            }
            
            config.geometry_zbound.clear();
            for (int i = 0; i < geometry.zbound_size(); i++) {
                config.geometry_zbound.push_back(geometry.zbound(i));
            }
            
            config.geometry_dbound.clear();
            for (int i = 0; i < geometry.dbound_size(); i++) {
                config.geometry_dbound.push_back(geometry.dbound(i));
            }
            
            config.geometry_image_width = geometry.image_width();
            config.geometry_image_height = geometry.image_height();
            config.geometry_feat_width = geometry.feat_width();
            config.geometry_feat_height = geometry.feat_height();
            config.geometry_num_camera = geometry.num_camera();
            
            config.geometry_dim.clear();
            for (int i = 0; i < geometry.geometry_dim_size(); i++) {
                config.geometry_dim.push_back(geometry.geometry_dim(i));
            }
        }
        
        // 后处理配置
        if (proto_config.has_postprocessor_config()) {
            const auto& postprocessor = proto_config.postprocessor_config();
            
            config.out_size_factor = postprocessor.out_size_factor();
            
            config.pc_range.clear();
            for (int i = 0; i < postprocessor.pc_range_size(); i++) {
                config.pc_range.push_back(postprocessor.pc_range(i));
            }
            
            config.post_center_range_start.clear();
            for (int i = 0; i < postprocessor.post_center_range_start_size(); i++) {
                config.post_center_range_start.push_back(postprocessor.post_center_range_start(i));
            }
            
            config.post_center_range_end.clear();
            for (int i = 0; i < postprocessor.post_center_range_end_size(); i++) {
                config.post_center_range_end.push_back(postprocessor.post_center_range_end(i));
            }
            
            config.postprocessor_voxel_size.clear();
            for (int i = 0; i < postprocessor.voxel_size_size(); i++) {
                config.postprocessor_voxel_size.push_back(postprocessor.voxel_size(i));
            }
            
            config.sorted_bboxes = postprocessor.sorted_bboxes();
        }
        
        // 模型路径配置
        if (proto_config.has_model_paths()) {
            const auto& paths = proto_config.model_paths();
            config.camera_backbone_path = paths.camera_backbone_path();
            config.lidar_backbone_path = paths.lidar_backbone_path();
            config.fuser_path = paths.fuser_path();
            config.head_bbox_path = paths.head_bbox_path();
            config.camera_vtransform_path = paths.camera_vtransform_path();
        }
        
        printf("Config loaded successfully from protobuf format.\n");
        return true;
        
    } catch (const std::exception& e) {
        printf("Error loading config file: %s\n", e.what());
        return false;
    }
}

// 创建BEVFusion核心的辅助函数 - 使用配置文件参数
std::shared_ptr<bevfusion::Core> BEVFusionAlgImplement::create_core_with_config(const BEVFusionConfig& config) {
    printf("Create BEVFusion core by %s, %s\n", config.model_name.c_str(), config.precision.c_str());

    // 1. 相机归一化参数配置 - 使用第一个相机的配置
    bevfusion::camera::NormalizationParameter normalization;
    if (!config.image_widths.empty()) {
        normalization.image_width = config.image_widths[0];
        normalization.image_height = config.image_heights[0];
        normalization.output_width = config.output_widths[0];
        normalization.output_height = config.output_heights[0];
        normalization.resize_lim = config.resize_lims[0];
    } else {
        // 默认值
        normalization.image_width = 1600;
        normalization.image_height = 900;
        normalization.output_width = 704;
        normalization.output_height = 256;
        normalization.resize_lim = 0.48f;
    }
    
    normalization.num_camera = config.num_camera;
    
    // 从配置文件读取插值方法
    if (config.interpolation == "Nearest") {
        normalization.interpolation = bevfusion::camera::Interpolation::Nearest;
    } else {
        // 默认使用 Bilinear
        normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;
    }

    // 使用配置文件中的归一化参数
    if (!config.means.empty() && !config.stds.empty() && !config.scale_factors.empty()) {
        float mean[3] = {config.means[0][0], config.means[0][1], config.means[0][2]};
        float std[3] = {config.stds[0][0], config.stds[0][1], config.stds[0][2]};
        normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, config.scale_factors[0], config.offsets[0]);
    } else {
        // 默认值
        float mean[3] = {0.485, 0.456, 0.406};
        float std[3] = {0.229, 0.224, 0.225};
        normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);
    }

    // 2. 激光雷达体素化参数配置
    bevfusion::lidar::VoxelizationParameter voxelization;
    if (config.lidar_min_range.size() >= 3 && config.lidar_max_range.size() >= 3) {
        voxelization.min_range = nvtype::Float3(config.lidar_min_range[0], config.lidar_min_range[1], config.lidar_min_range[2]);
        voxelization.max_range = nvtype::Float3(config.lidar_max_range[0], config.lidar_max_range[1], config.lidar_max_range[2]);
    } else {
        voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
        voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
    }
    
    if (config.lidar_voxel_size.size() >= 3) {
        voxelization.voxel_size = nvtype::Float3(config.lidar_voxel_size[0], config.lidar_voxel_size[1], config.lidar_voxel_size[2]);
    } else {
        voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
    }
    
    voxelization.grid_size = voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
    voxelization.max_points_per_voxel = config.max_points_per_voxel;
    voxelization.max_points = config.max_points;
    voxelization.max_voxels = config.max_voxels;
    voxelization.num_feature = config.num_feature;

    // 3. 激光雷达 SCN 参数配置
    bevfusion::lidar::SCNParameter scn;
    scn.voxelization = voxelization;
    scn.model = config.lidar_model_path;
    scn.order = (config.coordinate_order == "XYZ") ? bevfusion::lidar::CoordinateOrder::XYZ : bevfusion::lidar::CoordinateOrder::XYZ;

    // 4. 精度设置
    if (config.precision == "int8") {
        scn.precision = bevfusion::lidar::Precision::Int8;
    } else {
        scn.precision = bevfusion::lidar::Precision::Float16;
    }

    // 5. 几何参数配置
    bevfusion::camera::GeometryParameter geometry;
    if (config.geometry_xbound.size() >= 3) {
        geometry.xbound = nvtype::Float3(config.geometry_xbound[0], config.geometry_xbound[1], config.geometry_xbound[2]);
    } else {
        geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    }
    
    if (config.geometry_ybound.size() >= 3) {
        geometry.ybound = nvtype::Float3(config.geometry_ybound[0], config.geometry_ybound[1], config.geometry_ybound[2]);
    } else {
        geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    }
    
    if (config.geometry_zbound.size() >= 3) {
        geometry.zbound = nvtype::Float3(config.geometry_zbound[0], config.geometry_zbound[1], config.geometry_zbound[2]);
    } else {
        geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
    }
    
    if (config.geometry_dbound.size() >= 3) {
        geometry.dbound = nvtype::Float3(config.geometry_dbound[0], config.geometry_dbound[1], config.geometry_dbound[2]);
    } else {
        geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
    }
    
    geometry.image_width = config.geometry_image_width;
    geometry.image_height = config.geometry_image_height;
    geometry.feat_width = config.geometry_feat_width;
    geometry.feat_height = config.geometry_feat_height;
    geometry.num_camera = config.geometry_num_camera;
    
    if (config.geometry_dim.size() >= 3) {
        geometry.geometry_dim = nvtype::Int3(config.geometry_dim[0], config.geometry_dim[1], config.geometry_dim[2]);
    } else {
        geometry.geometry_dim = nvtype::Int3(360, 360, 80);
    }

    // 6. 边界框转换参数配置
    bevfusion::head::transbbox::TransBBoxParameter transbbox;
    transbbox.out_size_factor = config.out_size_factor;
    
    if (config.pc_range.size() >= 2) {
        transbbox.pc_range = {config.pc_range[0], config.pc_range[1]};
    } else {
        transbbox.pc_range = {-54.0f, -54.0f};
    }
    
    if (config.post_center_range_start.size() >= 3) {
        transbbox.post_center_range_start = {config.post_center_range_start[0], config.post_center_range_start[1], config.post_center_range_start[2]};
    } else {
        transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
    }
    
    if (config.post_center_range_end.size() >= 3) {
        transbbox.post_center_range_end = {config.post_center_range_end[0], config.post_center_range_end[1], config.post_center_range_end[2]};
    } else {
        transbbox.post_center_range_end = {61.2, 61.2, 10.0};
    }
    
    if (config.postprocessor_voxel_size.size() >= 2) {
        transbbox.voxel_size = {config.postprocessor_voxel_size[0], config.postprocessor_voxel_size[1]};
    } else {
        transbbox.voxel_size = {0.075, 0.075};
    }
    
    transbbox.model = config.head_bbox_path;
    transbbox.confidence_threshold = config.confidence_threshold;
    transbbox.sorted_bboxes = config.sorted_bboxes;

    // 7. 核心参数配置
    bevfusion::CoreParameter param;
    param.camera_model = config.camera_backbone_path;
    param.normalize = normalization;
    param.lidar_scn = scn;
    param.geometry = geometry;
    param.transfusion = config.fuser_path;
    param.transbbox = transbbox;
    param.camera_vtransform = config.camera_vtransform_path;
    
    return bevfusion::create_core(param);
}

// 创建BEVFusion算法对象的工厂函数
extern "C" __attribute__ ((visibility("default"))) IBEVFusionAlg* CreateBEVFusionAlgObj(const std::string& p_strExePath) {
    (void)p_strExePath; // 路径在 initAlgorithm 中处理
    return new BEVFusionAlgImplement();
}
