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
from diffpose.ljubljana import Evaluator, LjubljanaDataset, Transforms
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor, SparseRegistration


def initialize(id_number, view, subsample=8):
    # Load the model
    ckpt = torch.load(f"checkpoints/specimen_{id_number:02d}_{view}_best.ckpt")
    model = PoseRegressor(
        ckpt["model_name"],
        ckpt["parameterization"],
        ckpt["convention"],
        norm_layer=ckpt["norm_layer"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda()
    model.eval()

    # Load the subject
    subject = LjubljanaDataset(view)
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
        pose,
        isocenter_pose,
    ) = subject[id_number]
    volume[volume < 1000] = 0.0
    isocenter_pose = isocenter_pose.cuda()
    evaluator = Evaluator(subject, id_number)

    # Initialize DRR
    height //= subsample
    width //= subsample
    delx *= subsample
    dely *= subsample
    drr = DRR(
        volume,
        spacing,
        focal_len / 2,
        height,
        delx,
        width,
        dely,
        x0,
        y0,
        reverse_x_axis=True,
    ).to("cuda")
    transforms = Transforms(height, width)

    # Get predicted pose and features
    pose = pose.to("cuda")
    img = transforms(img).to("cuda")
    with torch.no_grad():
        offset = model(img)
        features = model.backbone.forward_features(img)
        features = resize(
            features,
            (height, width),
            interpolation=3,
            antialias=True,
        )
        features = features.sum(dim=[0, 1], keepdim=True)
        features -= features.min()
        features /= features.max() - features.min()
        features /= features.sum()
    pred_pose = isocenter_pose.compose(offset)

    return drr, img, pose, pred_pose, features, evaluator


class Registration:
    def __init__(
        self,
        drr,
        img,
        pose,
        pred_pose,
        features,
        evaluator,
        parameterization,
        convention="ZYX",
        lr_rot=1e-3,
        lr_xyz=1e0,
        n_iters=5000,
        verbose=True,
    ):
        self.parameterization = parameterization
        self.convention = convention
        self.registration, self.optimizer, self.scheduler = self.initialize(
            drr, pred_pose, features, lr_rot, lr_xyz
        )

        self.img = img
        self.pose = pose

        self.geodesics = GeodesicSE3()
        self.doublegeo = DoubleGeodesic(drr.detector.sdr)
        self.criterion = MultiscaleNormalizedCrossCorrelation2d(
            [None, 13],  # None means global NCC
            [0.5, 0.5],
        )
        self.target_registration_error = evaluator

        self.n_iters = n_iters
        self.verbose = verbose

    def initialize(self, drr, pred_pose, features, lr_rot, lr_xyz):
        registration = SparseRegistration(
            drr,
            pose=pred_pose,
            parameterization=self.parameterization,
            convention=self.convention,
            features=features,
        )
        optimizer = torch.optim.Adam(
            [
                {"params": [registration.rotation], "lr": lr_rot},
                {"params": [registration.translation], "lr": lr_xyz},
            ],
            maximize=True,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=500,
            gamma=0.9,
        )
        return registration, optimizer, scheduler

    def evaluate(self):
        est_pose = self.registration.get_current_pose()
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

    def run(self):
        # Initial loss
        param, geo, tre = self.evaluate()

        params = [param]
        losses = []
        geodesic = [geo]
        fiducial = [tre]
        times = []

        # Optimization loop
        itr = tqdm(range(self.n_iters), ncols=75) if self.verbose else range(self.n_iters)
        for _ in itr:
            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            pred_img, mask = self.registration()
            loss = self.criterion(pred_img, self.img)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            t1 = time.perf_counter()

            param, geo, tre = self.evaluate()
            params.append(param)
            losses.append(loss.item())
            geodesic.append(geo)
            fiducial.append(tre)
            times.append(t1 - t0)

        # Final pass
        pred_img, mask = self.registration()
        loss = self.criterion(pred_img, self.img)
        losses.append(loss.item())
        times.append(0)

        # Final dataframe
        df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
        df["ncc"] = losses
        df[["geo_r", "geo_t", "geo_d", "geo_se3"]] = geodesic
        df["fiducial"] = fiducial
        df["time"] = times
        df["parameterization"] = self.parameterization

        return df, self.registration.get_current_pose()


def main(id_number, view, parameterization="se3_log_map"):
    drr, img, pose, pred_pose, features, evaluator = initialize(id_number, view)

    # Save initial pose
    initial_pose_mat = pred_pose.as_matrix().cpu().numpy()
    with open(f"runs/specimen{id_number:02d}_{view}_initial_pose.pkl", "wb") as f:
        pickle.dump(initial_pose_mat, f)

    registration = Registration(
        drr, img, pose, pred_pose, features, evaluator, parameterization
    )
    df, final_pose = registration.run()

    # Save final pose
    final_pose_mat = final_pose.as_matrix().cpu().numpy()
    with open(f"runs/specimen{id_number:02d}_{view}_final_pose.pkl", "wb") as f:
        pickle.dump(final_pose_mat, f)

    df.to_csv(
        f"runs/specimen{id_number:02d}_{view}_{parameterization}.csv",
        index=False,
    )


if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    id_numbers = list(range(10))
    views = ["ap", "lat"]

    Path("runs").mkdir(exist_ok=True)

    for id_number, view in product(id_numbers, views):
        print(f"Running specimen {id_number:02d}, view {view}")
        main(id_number, view)
