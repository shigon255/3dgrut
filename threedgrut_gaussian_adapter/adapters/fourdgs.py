import numpy as np
import torch

from threedgrut_gaussian_adapter.camera import CameraState
from threedgrut_gaussian_adapter.gaussian_state import GaussianState
from threedgrut_gaussian_adapter.training import compute_grut_densification_stats


def gaussian_state_from_fourdgs(model, viewpoint_camera, stage="fine", time_value=None):
    means3d = model.get_xyz
    opacity = model._opacity
    shs = model.get_features
    scales = model._scaling
    rotations = model._rotation

    if time_value is None:
        if isinstance(viewpoint_camera, dict):
            time_value = viewpoint_camera["time"]
        else:
            time_value = viewpoint_camera.time
    time = torch.tensor(time_value).to(means3d.device).repeat(means3d.shape[0], 1)

    if "coarse" in stage:
        means_final, scales_final, rotations_final, opacity_final, shs_final = (
            means3d,
            scales,
            rotations,
            opacity,
            shs,
        )
    elif "fine" in stage:
        means_final, scales_final, rotations_final, opacity_final, shs_final = model._deformation(
            means3d,
            scales,
            rotations,
            opacity,
            shs,
            time,
        )
    else:
        raise NotImplementedError(f"Unsupported 4DGS stage: {stage}")

    return GaussianState(
        positions=means_final,
        rotations=model.rotation_activation(rotations_final),
        scales=model.scaling_activation(scales_final),
        densities=model.opacity_activation(opacity_final),
        features=shs_final.reshape(shs_final.shape[0], -1),
        active_sh_degree=model.active_sh_degree,
    )


def camera_state_from_fourdgs(camera):
    if isinstance(camera, dict):
        raise TypeError("PanopticSports dict cameras are not supported by the generic 4DGS adapter.")
    w2c = np.zeros((4, 4), dtype=np.float32)
    w2c[:3, 3] = camera.T
    w2c[:3, :3] = camera.R.transpose()
    w2c[3, 3] = 1.0
    c2w = torch.tensor(np.linalg.inv(w2c), dtype=torch.float32)
    last_c2w = None
    if getattr(camera, "last_camera", None) is not None:
        last_w2c = np.zeros((4, 4), dtype=np.float32)
        last_w2c[:3, 3] = camera.last_camera.T
        last_w2c[:3, :3] = camera.last_camera.R.transpose()
        last_w2c[3, 3] = 1.0
        last_c2w = torch.tensor(np.linalg.inv(last_w2c), dtype=torch.float32)
    return CameraState(
        width=int(camera.image_width),
        height=int(camera.image_height),
        fovx=float(camera.FoVx),
        fovy=float(camera.FoVy),
        c2w=c2w,
        image=getattr(camera, "original_image", None),
        mask=getattr(camera, "mask", None),
        time=float(camera.time),
        last_c2w=last_c2w,
    )


def apply_grut_densification_stats(model, xyz_final, cam_pos):
    mask, values = compute_grut_densification_stats(
        model._xyz,
        xyz_final,
        model._xyz.grad,
        cam_pos,
    )
    model.xyz_gradient_accum[mask] += values[mask]
    model.denom[mask] += 1
