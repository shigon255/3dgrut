import torch


def compute_grut_densification_stats(xyz_canonical, xyz_final, xyz_grad, cam_pos):
    if xyz_grad is None:
        raise ValueError("xyz_grad is required for 3DGRUT densification stats.")
    if xyz_final.shape != xyz_canonical.shape:
        raise ValueError(f"xyz_final shape {xyz_final.shape} does not match canonical shape {xyz_canonical.shape}.")
    cam_pos = cam_pos.to(xyz_canonical.device)
    mask = (xyz_grad != 0).max(dim=1)[0]
    distance_to_camera = (xyz_final[mask] - cam_pos).norm(dim=1, keepdim=True)
    values = torch.zeros((xyz_canonical.shape[0], 1), dtype=xyz_canonical.dtype, device=xyz_canonical.device)
    values[mask] = torch.norm(xyz_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2.0
    return mask, values
