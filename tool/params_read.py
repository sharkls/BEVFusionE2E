#!/usr/bin/env python3
import os
import numpy as np

# 需要读取的文件
base = "/share/Code/Lidar_AI_Solution/CUDA-BEVFusionv2/example-data"
files = [
    "camera_intrinsics.tensor",   # 期望 6x(3x3) 或 6x9
    "camera2lidar.tensor",        # 期望 6x(4x4) 或 6x16
    "lidar2image.tensor",         # 期望 6x(3x4) 或 6x12
    "img_aug_matrix.tensor"       # 期望 6x(3x3) 或 6x9
]

# 可切换的数据类型（如果输出看起来不合理，试试 float16）
DTYPE = np.float32  # 或者改成 np.float16
NUM_CAM = 6

def parse_intrinsics_from_4x4(arr, discard_head=7, block=16):
    # 按需求：丢弃前 discard_head 个数；之后每 16 个组成一个 4x4 矩阵
    if arr.size <= discard_head:
        raise ValueError("intrinsics tensor size too small")
    data = arr[discard_head:]
    usable = (data.size // block) * block
    data = data[:usable]
    mats4x4 = data.reshape(-1, 4, 4)
    # 取左上角 3x3 作为内参
    K_list = [m[:3, :3].copy() for m in mats4x4]
    # 只保留前 NUM_CAM 个
    return K_list[:NUM_CAM]

def parse_camera2lidar_from_4x4(arr, discard_head=7, num_cams=NUM_CAM):
    if arr.size <= discard_head:
        raise ValueError("camera2lidar tensor size too small")
    data = arr[discard_head:]
    need = num_cams * 16
    if data.size < need:
        raise ValueError(f"camera2lidar size not enough: have {data.size}, need {need}")
    data = data[:need]
    return data.reshape(num_cams, 4, 4)

def parse_lidar2image_from_3x4(arr, discard_head=7, num_cams=NUM_CAM):
    # 丢弃前 discard_head 个数，取后续 num_cams*12 个数并重排为 6 个 3x4
    if arr.size <= discard_head:
        raise ValueError("lidar2image tensor size too small")
    data = arr[discard_head:]
    need = num_cams * 12
    if data.size < need:
        raise ValueError(f"lidar2image size not enough: have {data.size}, need {need}")
    data = data[:need]
    return data.reshape(num_cams, 3, 4)

def parse_as_4x4_blocks(arr, discard_head=7):
    # 通用：丢弃前 discard_head 个数，余下按 4x4 连续分块打印
    if arr.size <= discard_head:
        return np.empty((0, 4, 4))
    data = arr[discard_head:]
    usable = (data.size // 16) * 16
    data = data[:usable]
    return data.reshape(-1, 4, 4)

def parse_img_aug_first9_from_16(arr, discard_head=7, num_cams=NUM_CAM):
    # 丢弃前 7 个，之后每 16 个为一组，取每组前 9 个作为 3x3 数据增强矩阵（行优先）
    if arr.size <= discard_head:
        return []
    data = arr[discard_head:]
    usable_groups = data.size // 16
    if usable_groups <= 0:
        return []
    data = data[:usable_groups * 16]
    groups = data.reshape(-1, 16)
    Ks = []
    for g in groups:
        first9 = g[:9]
        Ks.append(first9.reshape(3, 3))
    return Ks[:num_cams]

def read_tensor(path, dtype):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def pretty_print(name, arr, group_hint=None):
    print(f"=== {name} ===")
    print(f"dtype: {arr.dtype}, count: {arr.size}, bytes: {arr.nbytes}")
    # 打印前 32 个数值以便快速预览
    preview_count = min(103, arr.size)
    print(f"head({preview_count}): {arr[:preview_count]}")
    # 如果给了分组提示，尝试reshape展示
    if group_hint is not None and arr.size % group_hint == 0:
        try:
            grouped = arr.reshape(-1, group_hint)
            print(f"shape guess: {grouped.shape}")
            # 打印前 2 组
            rows = min(2, grouped.shape[0])
            print(f"first {rows} rows:")
            print(grouped[:rows])
        except Exception as e:
            print(f"(reshape failed: {e})")
    print()

def main():
    for fname in files:
        path = os.path.join(base, fname)
        if not os.path.isfile(path):
            print(f"[WARN] not found: {path}")
            continue

        arr = read_tensor(path, DTYPE)

        # 根据常见 BEVFusion 配置尝试给出 reshape 提示
        if "intrinsics" in fname:
            # 专按用户规则：跳7、每16成4x4，取左上3x3
            Ks = parse_intrinsics_from_4x4(arr, discard_head=7, block=16)
            print(f"=== {fname} (parsed by discard=7 then 4x4 -> 3x3) ===")
            for i, K in enumerate(Ks):
                flat = K.reshape(-1)
                vals = ", ".join(f"{v:.7g}" for v in flat)
                print(f"cam[{i}] intrinsics: [{vals}]")
            print()
        elif "camera2lidar" in fname:
            mats = parse_camera2lidar_from_4x4(arr, discard_head=7, num_cams=NUM_CAM)
            print(f"=== {fname} (parsed by discard=7 then 6x 4x4) ===")
            for i, M in enumerate(mats):
                flat = M.reshape(-1)
                vals = ", ".join(f"{v:.7g}" for v in flat)
                print(f"cam[{i}] camera2lidar(4x4): [{vals}]")
            print()
        elif "lidar2image" in fname:
            # 改为：丢弃7个后，余下整体按 4x4 分块排列
            mats = parse_as_4x4_blocks(arr, discard_head=7)
            print(f"=== {fname} (parsed by discard=7 then Nx 4x4) ===")
            for i, M in enumerate(mats):
                flat = M.reshape(-1)
                vals = ", ".join(f"{v:.7g}" for v in flat)
                print(f"block[{i}] 4x4: [{vals}]")
            print()
        elif "img_aug_matrix" in fname:
            # 先打印原始未处理的完整数据
            print(f"=== {fname} (RAW, no discard, full) ===")
            raw_vals = ", ".join(f"{v:.7g}" for v in arr.astype(float))
            print(f"count={arr.size}\n[{raw_vals}]\n")

            # 按规则：丢弃前7个，每16取前9个重组为3x3
            Ks = parse_img_aug_first9_from_16(arr, discard_head=7, num_cams=NUM_CAM)
            print(f"=== {fname} (parsed by discard=7 then take first 9 of each 16 -> 3x3) ===")
            for i, K in enumerate(Ks):
                flat = K.reshape(-1)
                vals = ", ".join(f"{v:.7g}" for v in flat)
                print(f"cam[{i}] img_aug_matrix(3x3): [{vals}]")
            print()
            # 同时打印完整的 4x4 分块（丢弃7个之后的全部数据）
            mats4 = parse_as_4x4_blocks(arr, discard_head=7)
            print(f"=== {fname} (full 4x4 blocks after discard=7) ===")
            for i, M in enumerate(mats4):
                flat = M.reshape(-1)
                vals = ", ".join(f"{v:.7g}" for v in flat)
                print(f"block[{i}] 4x4: [{vals}]")
            print()
        else:
            pretty_print(fname, arr)

if __name__ == "__main__":
    main()