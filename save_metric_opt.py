import os
import numpy as np
import pandas as pd
import pickle
import torch
from diffpose.calibration import RigidTransform

save_dir = '/lpai/volumes/lmm-data-proc/yuhao/code/DiffPose/result_diffpose'
folder_dir = "experiments/deepfluoro/output"

specimen_id_list = ["specimen_1", "specimen_2", "specimen_4", "specimen_5", "specimen_6"]

# 用于存储所有 specimen 的 pose 矩阵
all_poses_dict = {}

for specimen_id in specimen_id_list:
    folder_path = os.path.join(folder_dir, specimen_id)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    csv_files = sorted(csv_files, key=lambda x: int(x.split('xray')[-1].split("_")[0]))
    
    min_fiducial_values = []
    pose_matrices = []  # 存储对应的 pose 矩阵

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # 找到中位数 fiducial 对应的行（更稳健的选择）
        median_fiducial = df['fiducial'].median()
        # 找到最接近中位数的行
        median_idx = (df['fiducial'] - median_fiducial).abs().idxmin()
        selected_fiducial = df.loc[median_idx, 'fiducial']
        min_fiducial_values.append(selected_fiducial)
        
        # 提取对应行的姿态参数
        alpha = df.loc[median_idx, 'alpha']
        beta = df.loc[median_idx, 'beta']
        gamma = df.loc[median_idx, 'gamma']
        bx = df.loc[median_idx, 'bx']
        by = df.loc[median_idx, 'by']
        bz = df.loc[median_idx, 'bz']
        
        # 转换为 pose 矩阵
        pose = RigidTransform(
            torch.tensor([[alpha, beta, gamma]]),
            torch.tensor([[bx, by, bz]]),
            parameterization="euler_angles",
            convention="ZYX"
        )
        pose_matrix = pose.get_matrix()[0].T.cpu().numpy()  # shape (4, 4)
        pose_matrices.append(pose_matrix)

    result_array = np.array(min_fiducial_values)
    pose_array = np.array(pose_matrices)  # shape (N, 4, 4)
    
    # 将当前 specimen 的 pose 列表添加到字典中
    all_poses_dict[specimen_id] = pose_matrices  # 保存为 list of numpy arrays

    print(f"{specimen_id}: min fiducials shape = {result_array.shape}")
    print(f"{specimen_id}: poses shape = {pose_array.shape}")

    # 保存最小 fiducial 值
    save_path = os.path.join(save_dir, f"diffpose_{specimen_id}_opt.npy")
    np.save(save_path, result_array)
    
    # 保存对应的 pose 矩阵 (单个 specimen)
    pose_save_path = os.path.join(save_dir, f"diffpose_{specimen_id}_opt_poses.npy")
    np.save(pose_save_path, pose_array)
    
    print(f"Saved to {save_path} and {pose_save_path}\n")

# 保存所有 specimen 的 pose 到一个大的 pickle 文件
all_poses_pickle_path = os.path.join(save_dir, "diffpose_all_opt_poses.pkl")
with open(all_poses_pickle_path, "wb") as f:
    pickle.dump(all_poses_dict, f)

print(f"\nAll poses saved to {all_poses_pickle_path}")
print(f"Dictionary keys: {list(all_poses_dict.keys())}")
for key, poses in all_poses_dict.items():
    print(f"  {key}: {len(poses)} poses, each shape {poses[0].shape}")
