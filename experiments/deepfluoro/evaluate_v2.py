from pathlib import Path

import pandas as pd
# import submitit
import torch
import pickle
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.registration import PoseRegressor
from diffpose.calibration import RigidTransform

def load_specimen(id_number, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)
    return specimen, isocenter_pose


def evaluate(specimen, pred_poses):
    rows = []
    for idx in tqdm(range(len(specimen)), ncols=100):
        target_registration_error = Evaluator(specimen, idx)
        pred_pose = pred_poses[idx]

        R = pred_pose[:3, :3]
        t = pred_pose[:3, 3]
        pred_pose = RigidTransform(
            R=torch.from_numpy(R),
            t=torch.from_numpy(t)
        )

        mtre = target_registration_error(pred_pose).item()
        rows.append((idx, mtre))
    return rows


def main():
    device = torch.device("cuda")

    all_rows = []  # 用于最终合并所有 subject

    with open("/nas2/home/yuhao/code/DiffPose/diffpose_deepfluoro_final_pose.pkl", "rb") as f:
        pred_poses = pickle.load(f)

    for subject_id in range(1, 7):
        print(f"\n=== Processing subject {subject_id} ===")

        specimen, isocenter_pose = load_specimen(subject_id, device)
        subject_key = f"specimen_{subject_id}"

        rows = evaluate(specimen, pred_poses[subject_key])

        # 将 subject_id 信息加入每一行
        for sample_id, error in rows:
            all_rows.append((subject_id, sample_id, error))

    # 构建统一 DataFrame
    df = pd.DataFrame(all_rows, columns=["subject_id", "sample_id", "error"])

    # 保存到一个csv
    df.to_csv("debug/all_subjects_errors.csv", index=False)

    print("\nSaved: debug/all_subjects_errors.csv")


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Path("evaluations").mkdir(exist_ok=True)
    # id_numbers = [1, 2, 3, 4, 5, 6]

    # executor = submitit.AutoExecutor(folder="logs")
    # executor.update_parameters(
    #     name="eval",
    #     gpus_per_node=1,
    #     mem_gb=10.0,
    #     slurm_array_parallelism=len(id_numbers),
    #     slurm_exclude="curcum",
    #     slurm_partition="2080ti",
    #     timeout_min=10_000,
    # )
    # jobs = executor.map_array(main, id_numbers)


    main()