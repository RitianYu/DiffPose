import pickle
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from diffpose.ljubljana import LjubljanaDataset

def main():
    """
    从LjubljanaDataset中提取所有specimen的ground truth pose
    保存格式:
    {
        "specimen_1": [pose_ap, pose_lat],
        "specimen_2": [pose_ap, pose_lat],
        ...
    }
    """
    
    views = ["ap", "lat"]
    n_specimens = 10  # Ljubljana dataset有10个specimen
    
    # 用于存储所有specimen的pose
    all_gt_poses = {}
    
    for specimen_id in range(1, n_specimens + 1):
        specimen_key = f"specimen_{specimen_id}"
        poses_for_specimen = []
        
        print(f"Processing {specimen_key}...")
        
        for view in views:
            # 加载对应view的dataset
            dataset = LjubljanaDataset(view)
            
            # 获取该specimen的数据 (id_number从0开始索引，但specimen从1开始编号)
            (
                volume,
                spacing,
                focal_len,
                height,
                width,
                delx,
                dely,
                x0,
                y0,
                img,
                pose,  # 这是ground truth pose
                isocenter_pose,
            ) = dataset[specimen_id - 1]
            
            # 将pose转换为4x4矩阵并转为numpy
            pose_matrix = pose.get_matrix()[0].cpu().numpy()  # shape (4, 4)
            poses_for_specimen.append(pose_matrix)
            
            print(f"  {view}: pose shape = {pose_matrix.shape}")
        
        # 保存该specimen的两个view的pose
        all_gt_poses[specimen_key] = poses_for_specimen
    
    # 保存到pickle文件
    save_path = "diffpose_lju_gt_poses.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_gt_poses, f)
    
    print(f"\n所有Ground Truth poses已保存到: {save_path}")
    print(f"字典包含的specimen: {list(all_gt_poses.keys())}")
    print(f"每个specimen有 {len(all_gt_poses['specimen_1'])} 个view (ap, lat)")
    print(f"每个pose的shape: {all_gt_poses['specimen_1'][0].shape}")

if __name__ == "__main__":
    main()
