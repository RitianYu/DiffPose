import os
import pickle
import re

INPUT_DIR = "./results/diffpose/lju"
OUTPUT_PKL = "diffpose_lju_final_pose.pkl"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    merged = {}

    for filename in sorted(os.listdir(INPUT_DIR)):  # 排序文件名
        if not filename.endswith(".pkl"):
            continue

        # 只匹配 final pose
        match = re.match(r"specimen(\d+)_(ap|lat)_final_pose\.pkl", filename)
        if not match:
            continue

        specimen_num, view = match.groups()
        specimen_key = f"specimen_{int(specimen_num)}"

        file_path = os.path.join(INPUT_DIR, filename)
        pose = load_pkl(file_path)

        # 初始化为 [None, None]
        if specimen_key not in merged:
            merged[specimen_key] = [None, None]

        if view == "ap":
            merged[specimen_key][0] = pose
        elif view == "lat":
            merged[specimen_key][1] = pose

        print(f"Loaded {filename} → {specimen_key} ({view})")

    # 检查是否有 specimen 的 pose 缺失
    for k, v in merged.items():
        if v[0] is None or v[1] is None:
            print(f"WARNING: {k} missing AP or LAT pose: {v}")

    # 保存最终 dict
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(merged, f)

    print("\nSaved to:", OUTPUT_PKL)
    print("Specimens:", list(merged.keys()))


if __name__ == "__main__":
    main()
