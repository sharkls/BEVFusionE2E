import numpy as np
# 可选：如需真正做去畸变，可引入 cv2 并在你的图像预处理里调用
# import cv2

def compute_bevfusion_matrices(
    K,
    dist_coeffs=None,
    T_ext=None,
    ext_type='lidar2cam',   # 'lidar2cam' 或 'cam2lidar'
    img_size=None,          # (img_w, img_h)，原始图像分辨率
    out_size=None,          # (out_w, out_h)，模型输入分辨率
    keep_ratio=True         # 是否保持宽高比并做居中裁剪
):
    """
    K:           3x3 相机内参矩阵 (numpy.array)
    dist_coeffs: 畸变系数，形如 [k1, k2, p1, p2, k3] 或 [k1, k2, k3]，本函数不写进矩阵，只保留入口
    T_ext:       4x4 外参矩阵：
                   - 若 ext_type='lidar2cam'，则 X_cam = T_ext * X_lidar
                   - 若 ext_type='cam2lidar'，则 X_lidar = T_ext * X_cam
    img_size:    (img_w, img_h) 原图大小，例如 (1920, 1080)
    out_size:    (out_w, out_h) 模型输入大小，例如 (704, 256)
    """

    K = np.asarray(K, dtype=np.float64)
    T_ext = np.asarray(T_ext, dtype=np.float64)

    # 1) intrinsics: 4x4 齐次内参矩阵
    K4 = np.eye(4, dtype=np.float64)
    K4[:3, :3] = K

    # 2) 处理外参方向，得到 T_lidar2cam 和 T_cam2lidar
    if ext_type == 'lidar2cam':
        T_lidar2cam = T_ext
        T_cam2lidar = np.linalg.inv(T_lidar2cam)
    elif ext_type == 'cam2lidar':
        T_cam2lidar = T_ext
        T_lidar2cam = np.linalg.inv(T_cam2lidar)
    else:
        raise ValueError("ext_type 必须是 'lidar2cam' 或 'cam2lidar'")

    # 3) lidar2image: P = K [R|t]，再嵌入 4x4
    R = T_lidar2cam[:3, :3]
    t = T_lidar2cam[:3, 3:4]      # 3x1
    Rt = np.concatenate([R, t], axis=1)   # 3x4
    P = K @ Rt                    # 3x4

    T_lidar2image = np.eye(4, dtype=np.float64)
    T_lidar2image[:3, :4] = P

    # 4) img_aug_matrix：根据 img_size 和 out_size 生成一个缩放+居中裁剪矩阵
    if img_size is not None and out_size is not None:
        img_w, img_h = img_size
        out_w, out_h = out_size

        if keep_ratio:
            # 以保持宽高比的方式缩放，并居中裁剪/填充
            scale = min(out_w / img_w, out_h / img_h)
            # 缩放后的图像尺寸
            scaled_w = img_w * scale
            scaled_h = img_h * scale
            # 为了居中，把缩放后的图像放到输出中心，计算平移量
            tx = (out_w - scaled_w) / 2.0
            ty = (out_h - scaled_h) / 2.0
            sx, sy = scale, scale
        else:
            # 直接非等比缩放到输出尺寸
            sx = out_w / img_w
            sy = out_h / img_h
            tx, ty = 0.0, 0.0

        img_aug = np.eye(4, dtype=np.float64)
        img_aug[0, 0] = sx
        img_aug[1, 1] = sy
        img_aug[0, 3] = tx
        img_aug[1, 3] = ty
    else:
        # 如果没有给尺寸信息，就给一个单位矩阵
        img_aug = np.eye(4, dtype=np.float64)

    # BEVFusion 配置里需要的是“按行展开的 16 个数”
    def to_row_major_list(M):
        return [float(v) for v in M.reshape(-1)]

    result = {
        "intrinsics_matrix": K4,
        "intrinsics_list": to_row_major_list(K4),

        "camera2lidar_matrix": T_cam2lidar,
        "camera2lidar_list": to_row_major_list(T_cam2lidar),

        "lidar2image_matrix": T_lidar2image,
        "lidar2image_list": to_row_major_list(T_lidar2image),

        "img_aug_matrix": img_aug,
        "img_aug_list": to_row_major_list(img_aug),
    }

    return result


if __name__ == "__main__":
    # ======= 示例：用你前面那组参数演示 =======
    # 1) 相机内参
    K = [
        [1595.14160156684, 0.0,                1130.11716505185],
        [0.0,              1600.28282041317,   605.424014930734],
        [0.0,              0.0,                1.0]
    ]

    # 2) 畸变系数（这里不参与矩阵计算，实际用在图像去畸变中）
    # 例如：k1, k2, p1, p2, k3 或者 [k1, k2, k3]
    radial = [-0.298267394293707, 0.115289974930856, -0.049725905080390]
    # 如果有切向畸变 p1, p2，可以也放进来：
    tangential = [0.0, 0.0]  # 这里只是示例
    dist_coeffs = radial + tangential

    # 3) 外参：假设你现在这 4x4 是 lidar -> cam
    T_lidar2cam = [
        [0.27114073, -0.96250377, -0.00831869,  0.33706108],
        [-0.11899267, -0.02494201, -0.99258181, -0.39568526],
        [0.95515624,  0.27011922, -0.12129369,  4.74995206],
        [0.0,         0.0,         0.0,         1.0]
    ]

    # 4) 图像尺寸与网络输入尺寸
    img_size = (1920, 1080)   # 示例：原图 1920x1080
    out_size = (704, 256)     # 示例：BEVFusion 模型输入 704x256

    mats = compute_bevfusion_matrices(
        K=K,
        dist_coeffs=dist_coeffs,
        T_ext=T_lidar2cam,
        ext_type='lidar2cam',
        img_size=img_size,
        out_size=out_size,
        keep_ratio=True
    )

    # 打印，方便直接粘到 BEVFusionAlgConfig.conf
    def fmt(lst):
        return ", ".join(f"{v:.9f}" for v in lst)

    print("intrinsics:", fmt(mats["intrinsics_list"]))
    print("\ncamera2lidar:", fmt(mats["camera2lidar_list"]))
    print("\nlidar2image:", fmt(mats["lidar2image_list"]))
    print("\nimg_aug_matrix:", fmt(mats["img_aug_list"]))