import os
import pickle
import re

INPUT_DIR = "./results/diffpose/deepfluoro"
OUTPUT_PKL = "diffpose_deepfluoro_final_pose.pkl"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    merged = {}

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".pkl"):
            continue

        # 匹配 final pose 文件
        match = re.match(r"(specimen)(\d+)_se3_log_map_final_poses\.pkl", filename)
        if not match:
            continue

        prefix, specimen_num = match.groups()
        specimen_key = f"specimen_{int(specimen_num)}"

        file_path = os.path.join(INPUT_DIR, filename)
        data_dict = load_pkl(file_path)

        # 原格式是 {0: pose, 1: pose, ...}
        # 转成 list：按 key 排序确保顺序一致
        pose_list = [data_dict[k] for k in sorted(data_dict.keys())]

        merged[specimen_key] = pose_list

        print(f"Loaded {filename} → {specimen_key} ({len(pose_list)} poses)")

    # 保存最终 dict
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(merged, f)

    print("\nAll specimens merged (dict → list)!")
    print("Saved to:", OUTPUT_PKL)
    print("Specimens:", list(merged.keys()))


if __name__ == "__main__":
    main()
