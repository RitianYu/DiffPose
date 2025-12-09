import pandas as pd

# 读取 CSV
df = pd.read_csv("/nas2/home/yuhao/code/DiffPose/experiments/deepfluoro/debug/all_subjects_errors.csv")   # 换成你的文件名

# 计算统计量
median_error = df["error"].median()
mean_error = df["error"].mean()

# success rate
sr_1mm = (df["error"] < 1).mean()
sr_3mm = (df["error"] < 3).mean()
sr_5mm = (df["error"] < 5).mean()
sr_10mm = (df["error"] < 10).mean()

print(f"Median error: {median_error:.4f} mm")
print(f"Mean error:   {mean_error:.4f} mm")
print(f"Success rate < 1mm:  {sr_1mm:.4f}")
print(f"Success rate < 3mm:  {sr_3mm:.4f}")
print(f"Success rate < 5mm:  {sr_5mm:.4f}")
print(f"Success rate < 10mm: {sr_10mm:.4f}")