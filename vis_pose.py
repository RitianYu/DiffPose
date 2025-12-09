import numpy as np
import pickle
import matplotlib.pyplot as plt
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms


def plot_pose(ax, T, color):
    if T.shape == (4, 4):
        pass
    else:
        T = T[0]
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    ax.scatter(x, y, z, color=color, s=30)
    # # 绘制三个坐标轴
    ax.quiver(x, y, z, *R[:,0], length=50, color='r')  # X-axis
    ax.quiver(x, y, z, *R[:,1], length=50, color='g')  # Y-axis
    ax.quiver(x, y, z, *R[:,2], length=50, color='b')  # Z-axis


with open("/nas2/home/yuhao/code/xvr/results/deepfluoro/deepfluoro_final_pose_dict.pkl", "rb") as f:
    data_final = pickle.load(f)  

final_poses = data_final['subject01']
specimen = DeepFluoroDataset(1, filename="/nas2/home/yuhao/code/xvr/data/ipcai_2020_full_res_data.h5")
isocenterpose = specimen.isocenter_pose.get_matrix().cpu().numpy()[0].T

gt_poses = []
for idx in range(len(specimen)):
    pose = specimen[idx][1].get_matrix().cpu().numpy()[0].T  # shape [4,4]
    gt_poses.append(pose)
gt_poses = np.stack(gt_poses)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, T in enumerate(final_poses[0::40]):
    plot_pose(ax, T, "#FFA600FF")
for i, T in enumerate(gt_poses[0::40]):
    plot_pose(ax, T, "#800080")


plot_pose(ax, isocenterpose, "#17D5E2")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Final Pose (org) vs GT Pose (purple)')
plt.savefig("poses_compare.png", dpi=200, bbox_inches='tight')
