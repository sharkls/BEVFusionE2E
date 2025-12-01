import numpy as np
import cv2


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
    dist_coeffs: 畸变系数，这里不参与矩阵计算，只保留入口
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
            scaled_w = img_w * scale
            scaled_h = img_h * scale
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

    # 展开成 BEVFusion 需要的 16 个数（行优先）
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


# ============ 用户可修改区：标定数据 & 相机参数 ============

def solve_pnp_and_build_extrinsic():
    """
    使用你的 object_points, image_points, K, dist_coeffs
    通过 solvePnP 计算 T_lidar2cam (或 T_world2cam) 4x4 外参矩阵。
    """

    # 1) 3D 点（激光雷达/世界坐标系下）
    object_points = np.array([
        [9.5767800,  2.5911900,  -0.688533],
        [25.345301, 31.624800,   4.102380],
        [31.069000, 15.093000,   2.682050],
        [37.104900, 11.308700,   2.787310],
        [33.615002,  0.299221,   4.712500],
        [29.401100, -6.340620,   3.086940],
    ], dtype=np.float32)

    # 2) 像素坐标
    image_points = np.array([
        [1180, 507],
        [250,  283],
        [876,  312],
        [1111, 321],
        [1509, 223],
        [1830, 272],
    ], dtype=np.float32)

    # 3) 相机内参矩阵
    camera_matrix = np.array([
        [1595.14160156684, 0.0,                1130.11716505185],
        [0.0,              1600.28282041317,   605.424014930734],
        [0.0,              0.0,                1.0]
    ], dtype=np.float32)

    # 4) 畸变系数 (K1, K2, P1, P2, K3, K4, K5, K6)
    # 这里只是你给的示例，可按实际标定结果修改
    dist_coeffs = np.array([-0.3225, 0.1668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           dtype=np.float32)

    # ---- 使用 solvePnP 计算外参 ----
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    print("solvePnP success:", success)
    if not success:
        raise RuntimeError("solvePnP failed to find a solution")

    # 旋转向量 -> 旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 构建 4x4 T_lidar2cam
    T_lidar2cam = np.eye(4, dtype=np.float64)
    T_lidar2cam[:3, :3] = rotation_matrix
    T_lidar2cam[:3, 3] = translation_vector.reshape(-1)

    print("\n=== T_lidar2cam (4x4) ===")
    print(T_lidar2cam)

    # 可选：计算投影误差，检查标定质量
    projected_points, _ = cv2.projectPoints(
        object_points, rotation_vector, translation_vector,
        camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(image_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    print("\nProjection Errors (pixels):", errors)
    print("Mean Projection Error (pixels):", mean_error)

    return camera_matrix, dist_coeffs, T_lidar2cam


# ============ 主流程：从 PnP -> BEVFusion 配置矩阵 ============

if __name__ == "__main__":
    # 1) 先用 solvePnP 得到外参 T_lidar2cam
    K, dist_coeffs, T_lidar2cam = solve_pnp_and_build_extrinsic()

    # 2) 图像尺寸 和 模型输入尺寸（按你的工程修改）
    img_size = (1920, 1080)   # 原始图像宽高
    out_size = (704, 256)     # BEVFusion 模型输入宽高

    mats = compute_bevfusion_matrices(
        K=K,
        dist_coeffs=dist_coeffs,
        T_ext=T_lidar2cam,
        ext_type='lidar2cam',  # 我们的 T_lidar2cam 符合 X_cam = T * X_lidar
        img_size=img_size,
        out_size=out_size,
        keep_ratio=True
    )

    def fmt(lst):
        return ", ".join(f"{v:.9f}" for v in lst)

    # ====== 计算与 cameras 配置块对应的其余字段 ======
    img_w, img_h = img_size
    out_w, out_h = out_size
    # 为了对齐官方 BEVFusion / nuScenes 配置风格，这里使用固定的 resize_lim
    # 如需调整，只修改这一行即可
    resize_lim = 0.48

    # 按当前工程默认的归一化参数（如需修改可在此处改）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale_factor = 0.003921568627451
    offset = 0.0
    interpolation = "Bilinear"

    # ====== 直接输出完整的 cameras { ... } 配置块，方便粘贴 ======
    print("\n================ BEVFusion cameras 配置片段（可直接粘贴） ================\n")
    print("cameras {")
    print("  # 自动生成的相机配置")
    print(f"  intrinsics: [{fmt(mats['intrinsics_list'])}] # 相机内参矩阵 (4x4矩阵，按行优先顺序排列)")
    print(f"  camera2lidar: [{fmt(mats['camera2lidar_list'])}] # 相机到激光雷达变换矩阵 (4x4矩阵，按行优先顺序排列)")
    print(f"  lidar2image: [{fmt(mats['lidar2image_list'])}] # 激光雷达到图像变换矩阵 (4x4矩阵，按行优先顺序排列)")
    print(f"  img_aug_matrix: [{fmt(mats['img_aug_list'])}] # 图像增强矩阵 (4x4矩阵，按行优先顺序排列)")
    print("")
    print("  # 相机归一化参数")
    print(f"  image_width: {img_w}")
    print(f"  image_height: {img_h}")
    print(f"  output_width: {out_w}")
    print(f"  output_height: {out_h}")
    print(f"  resize_lim: {resize_lim:.6f}")
    print("")
    print("  # 图像归一化参数")
    print(f"  mean: [{', '.join(f'{v:.3f}' for v in mean)}]")
    print(f"  std: [{', '.join(f'{v:.3f}' for v in std)}]")
    print(f"  scale_factor: {scale_factor}")
    print(f"  offset: {offset}")
    print(f"  interpolation: \"{interpolation}\"  # 图像插值方法：\"Bilinear\" 或 \"Nearest\"")
    print("}")