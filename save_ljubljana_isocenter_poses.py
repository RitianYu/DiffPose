import pickle
import numpy as np
import torch
from diffpose.ljubljana import LjubljanaDataset

# 定义所有 views
views = ["ap", "lat"]

# 存储所有 isocenter poses 的字典
all_isocenter_poses = {}

for view in views:
    print(f"Processing view: {view}")
    
    # 加载数据集
    dataset = LjubljanaDataset(view=view)
    
    # 存储当前 view 的所有 isocenter poses
    view_poses = []
    
    for idx in range(len(dataset)):
        # 获取数据
        (volume, spacing, focal_len, height, width, delx, dely, 
         x0, y0, img, pose, isocenter_pose) = dataset[idx]
        
        # 将 isocenter_pose 转换为 4x4 numpy 矩阵
        isocenter_matrix = isocenter_pose.get_matrix()[0].T.cpu().numpy()
        view_poses.append(isocenter_matrix)
        
        print(f"  idx {idx}: isocenter_pose shape = {isocenter_matrix.shape}")
        
        # 特别打印 lat view 下 subject_id 1, 2, 6 的 isocenter pose 值
        if view == "lat" and idx in [0, 1, 5]:  # idx 0, 1, 5 对应 subject 1, 2, 6
            subject_id = idx + 1
            print(f"\n  === Subject {subject_id} (lat view) isocenter_pose ===")
            print(isocenter_matrix)
            print()
    
    # 将当前 view 的 poses 存入字典
    all_isocenter_poses[view] = view_poses
    print(f"  Total {len(view_poses)} poses for view '{view}'\n")

# 保存为 pickle 文件
output_path = "ljubljana_isocenter_poses.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_isocenter_poses, f)

print(f"All isocenter poses saved to {output_path}")
print(f"\nSummary:")
for view, poses in all_isocenter_poses.items():
    print(f"  {view}: {len(poses)} poses, each shape {poses[0].shape}")

# 也可以选择保存为 numpy 格式
for view, poses in all_isocenter_poses.items():
    poses_array = np.array(poses)
    numpy_path = f"ljubljana_isocenter_poses_{view}.npy"
    np.save(numpy_path, poses_array)
    print(f"  Saved {view} to {numpy_path} with shape {poses_array.shape}")
