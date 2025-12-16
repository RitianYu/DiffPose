import pickle
import numpy as np
import torch
from diffpose.deepfluoro import DeepFluoroDataset

# 定义所有 specimen IDs
specimen_ids = [1, 2, 3, 4, 5, 6]

# 存储所有 isocenter poses 的字典
all_isocenter_poses = {}

for specimen_id in specimen_ids:
    print(f"Processing specimen {specimen_id}")
    
    # 加载数据集
    dataset = DeepFluoroDataset(specimen_id)
    
    # 获取 isocenter pose (每个 specimen 只有一个 isocenter pose)
    isocenter_pose = dataset.isocenter_pose
    
    # 将 isocenter_pose 转换为 4x4 numpy 矩阵
    isocenter_matrix = isocenter_pose.get_matrix()[0].T.cpu().numpy()
    
    # 存入字典
    all_isocenter_poses[f"specimen_{specimen_id}"] = isocenter_matrix
    
    print(f"  isocenter_pose shape = {isocenter_matrix.shape}")
    
    # 特别打印 specimen 4 的 isocenter pose 值
    if specimen_id == 4:
        print(f"\n  === Specimen 4 isocenter_pose ===")
        print(isocenter_matrix)
        print()

print("\n" + "="*60)
print("Summary of all specimens:")
print("="*60)

# 保存为 pickle 文件
output_path = "deepfluoro_isocenter_poses.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_isocenter_poses, f)

print(f"\nAll isocenter poses saved to {output_path}")
print(f"\nDictionary keys: {list(all_isocenter_poses.keys())}")
for key, pose in all_isocenter_poses.items():
    print(f"  {key}: shape {pose.shape}")

# 也可以选择保存为 numpy 格式 (所有 specimen 堆叠在一起)
poses_array = np.stack([all_isocenter_poses[f"specimen_{i}"] for i in specimen_ids])
numpy_path = "deepfluoro_isocenter_poses.npy"
np.save(numpy_path, poses_array)
print(f"\nAll poses stacked and saved to {numpy_path} with shape {poses_array.shape}")

# 额外单独保存 specimen 4
specimen_4_path = "deepfluoro_specimen_4_isocenter_pose.npy"
np.save(specimen_4_path, all_isocenter_poses["specimen_4"])
print(f"Specimen 4 pose saved to {specimen_4_path}")
