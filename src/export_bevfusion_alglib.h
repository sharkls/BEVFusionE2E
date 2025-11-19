/*******************************************************
 文件名：export_bevfusion_alglib.h
 作者：sharkls
 描述：BEVFusion算法库的算法接口类导出函数头文件
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#pragma once
#include <string>
#include <vector>
#include "common/global_context.h"
#include "common/dtype.hpp"
#include "CAlgResult.h"

// BEVFusion算法接口类
struct IBEVFusionAlg
{
    IBEVFusionAlg(){};
    virtual ~IBEVFusionAlg(){};

    // 初始化算法接口对象，内部主要处理只需初始化一次的操作，比如模型加载之类的，成功返回true，失败返回false
    virtual bool initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd) = 0;

    // 执行算法函数，传入原始数据体，算法执行成功返回处理后的数据或者检测结果（由算法类型而定），失败返回nullptr
    virtual void runAlgorithm(void* p_pSrcData) = 0;
};

// 输入数据结构 - 统一在头文件中定义
struct BEVFusionInputData {
    std::vector<unsigned char*> images;
    nvtype::half* lidar_points;
    int num_points;
};

// 创建BEVFusion算法对象的工厂函数
extern "C" __attribute__ ((visibility("default"))) IBEVFusionAlg* CreateBEVFusionAlgObj(const std::string& p_strExePath);