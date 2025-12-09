from diffpose.deepfluoro import DeepFluoroDataset
from diffpose.calibration import RigidTransform
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
import pickle
import torch
import numpy as np

# Load saved poses
with open("/nas2/home/yuhao/code/xvr/results/deepfluoro/deepfluoro_init_pose_dict.pkl", "rb") as f:
    data_init = pickle.load(f)
with open("/nas2/home/yuhao/code/xvr/results/deepfluoro/deepfluoro_final_pose_dict.pkl", "rb") as f:
    data_final = pickle.load(f)

# Load DeepFluoro GT
specimen1 = DeepFluoroDataset(1)
isocenter_pose = specimen1.isocenter_pose
gt_pose = specimen1[1][1].get_matrix().numpy()[0].T
breakpoint()
init_pose  = data_init["subject01"][1][0]
final_pose = data_final["subject01"][1][0]
R = final_pose[:3, :3]
t = final_pose[:3, 3]
final_pose_compose_iso = isocenter_pose.compose(RigidTransform(torch.from_numpy(R), torch.from_numpy(t)))
breakpoint()

