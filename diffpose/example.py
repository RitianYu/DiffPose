from diffdrr.detector import make_xrays
from diffdrr.drr import DRR
from diffdrr.siddon import siddon_raycast
from deepfluoro import DeepFluoroDataset
from diffdrr.visualization import plot_drr
import torch
from matplotlib import pyplot as plt

specimen = DeepFluoroDataset(1, filename="/nas2/home/yuhao/code/xvr/data/ipcai_2020_full_res_data.h5")
drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.focal_len / 2,
        height=1436,
        delx=0.194,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
        bone_attenuation_multiplier=2.5,
    )

rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]])
xyz = (torch.tensor(specimen.volume.shape) * specimen.spacing / 2).unsqueeze(0).float()
img = drr(rot, xyz, parameterization="euler_angles", convention="ZXY")

# gt_pose = specimen[0][1]
# breakpoint()
# img = drr(gt_pose.get_rotation(), gt_pose.get_translation(), parameterization="matrix")
plot_drr(img, ticks=False)
plt.savefig("drr_output_isocenter.png", dpi=300, bbox_inches="tight")
plt.close()