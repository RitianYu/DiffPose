import time
from itertools import product
from pathlib import Path
import pickle
import pandas as pd
import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from torchvision.transforms.functional import resize
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from diffpose.calibration import RigidTransform, convert
from diffpose.deepfluoro import DeepFluoroDataset, Evaluator, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor, SparseRegistration


class Registration:
    def __init__(
        self,
        drr,
        specimen,
        model,
        parameterization,
        convention=None,
        n_iters=500,
        verbose=False,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.drr = drr.to(self.device)
        self.model = model.to(self.device)
        model.eval()

        self.specimen = specimen
        self.isocenter_pose = specimen.isocenter_pose.to(self.device)

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(sdr=self.specimen.focal_len / 2)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d([None, 13], [0.5, 0.5])
        self.transforms = Transforms(self.drr.detector.height)
        self.parameterization = parameterization
        self.convention = convention

        self.n_iters = n_iters
        self.verbose = verbose

    def initialize_registration(self, img):
        with torch.no_grad():
            offset = self.model(img)
            features = self.model.backbone.forward_features(img)
            features = resize(
                features,
                (self.drr.detector.height, self.drr.detector.width),
                interpolation=3,
                antialias=True,
            )
            features = features.sum(dim=[0, 1], keepdim=True)
            features -= features.min()
            features /= features.max() - features.min()
            features /= features.sum()

        pred_pose = self.isocenter_pose.compose(offset)

        return SparseRegistration(
            self.drr,
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        ), pred_pose

    def initialize_optimizer(self, registration):
        optimizer = torch.optim.Adam(
            [
                {"params": [registration.rotation], "lr": 7.5e-3},
                {"params": [registration.translation], "lr": 7.5e0},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=25,
            gamma=0.9,
        )
        return optimizer, scheduler

    def evaluate(self, registration):
        est_pose = registration.get_current_pose()
        rot = est_pose.get_rotation("euler_angles", "ZYX")
        xyz = est_pose.get_translation()
        alpha, beta, gamma = rot.squeeze().tolist()
        bx, by, bz = xyz.squeeze().tolist()
        param = [alpha, beta, gamma, bx, by, bz]
        geo = (
            torch.concat(
                [
                    *self.doublegeo(est_pose, self.pose),
                    self.geodesics(est_pose, self.pose),
                ]
            )
            .squeeze()
            .tolist()
        )
        tre = self.target_registration_error(est_pose.cpu()).item()
        return param, geo, tre

    def run(self, idx):
        img, pose = self.specimen[idx]
        img = self.transforms(img).to(self.device)
        self.pose = pose.to(self.device)

        # -------------- Initial Pose ------------------
        registration, initial_pose = self.initialize_registration(img)

        optimizer, scheduler = self.initialize_optimizer(registration)
        self.target_registration_error = Evaluator(self.specimen, idx)

        params = []
        losses = []
        geodesic = []
        fiducial = []
        times = []

        param, geo, tre = self.evaluate(registration)
        params.append(param)
        geodesic.append(geo)
        fiducial.append(tre)

        itr = tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        for _ in itr:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            pred_img, mask = registration()
            loss = self.criterion(pred_img, img)
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.perf_counter()

            param, geo, tre = self.evaluate(registration)
            params.append(param)
            losses.append(loss.item())
            geodesic.append(geo)
            fiducial.append(tre)
            times.append(t1 - t0)

        # final loss
        pred_img, mask = registration()
        loss = self.criterion(pred_img, img)
        losses.append(loss.item())
        times.append(0)

        # -------------- Final pose -------------------
        final_pose = registration.get_current_pose()

        # Write results to dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        df["fiducial"] = fiducial
        df["time"] = times
        df["idx"] = idx
        df["parameterization"] = self.parameterization

        return df, initial_pose, final_pose


def main(id_number, parameterization):
    ckpt = torch.load(f"checkpoints/specimen_{id_number:02d}_best.ckpt")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    specimen = DeepFluoroDataset(id_number)
    height = ckpt["height"]
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=height,
        delx=delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )

    registration = Registration(
        drr,
        specimen,
        model,
        parameterization,
    )

    # 存储所有样本 pose
    initial_poses = {}
    final_poses = {}

    for idx in tqdm(range(len(specimen)), ncols=100):
        df, initial_pose, final_pose = registration.run(idx)

        df.to_csv(
            f"runs/specimen{id_number:02d}_xray{idx:03d}_{parameterization}.csv",
            index=False,
        )

        # 保存矩阵形式的 SE3
        initial_poses[idx] = initial_pose.as_matrix().cpu().numpy()
        final_poses[idx] = final_pose.as_matrix().cpu().numpy()

    # ----- 输出 pkl 文件 -----
    with open(f"runs/specimen{id_number:02d}_{parameterization}_initial_poses.pkl", "wb") as f:
        pickle.dump(initial_poses, f)

    with open(f"runs/specimen{id_number:02d}_{parameterization}_final_poses.pkl", "wb") as f:
        pickle.dump(final_poses, f)


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    id_numbers = [1, 2, 3, 4, 5, 6]
    parameterizations = [
        "se3_log_map",
        # "so3_log_map",
        # "axis_angle",
        # "euler_angles",
        # "quaternion",
        # "rotation_6d",
        # "rotation_10d",
        # "quaternion_adjugate",
    ]

    Path("runs").mkdir(exist_ok=True)

    for id_number, p in product(id_numbers, parameterizations):
        print(f"Running specimen {id_number}, parameterization {p}")
        main(id_number, p)
