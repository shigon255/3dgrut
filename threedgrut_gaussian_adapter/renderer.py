import math

import torch

from .camera import CameraState
from .camera_models import camera_state_to_batch, mask_batch
from .effects import RenderEffects, RollingShutterConfig
from .gaussian_state import GaussianState


def create_tracer(mode, config):
    if mode == "3dgrt":
        import threedgrt_tracer

        return threedgrt_tracer.Tracer(config)
    if mode == "3dgut":
        import threedgut_tracer

        return threedgut_tracer.Tracer(config)
    raise ValueError(f"Unsupported 3DGRUT mode: {mode}")


def render_gaussian_state(
    tracer,
    gaussians: GaussianState,
    camera: CameraState,
    effects: RenderEffects | None = None,
    bg_color: torch.Tensor | None = None,
    rebuild=True,
):
    gaussians.validate()
    effects = RenderEffects() if effects is None else effects
    batch = camera_state_to_batch(
        camera,
        effects,
        device=gaussians.positions.device,
        use_rolling_shutter=False,
    )
    tracer.build_acc_custom(
        positions=gaussians.positions,
        rotations=gaussians.rotations,
        scales=gaussians.scales,
        densities=gaussians.densities,
        rebuild=rebuild,
    )
    render_pkg = tracer.render_custom(
        positions=gaussians.positions,
        rotations=gaussians.rotations,
        scales=gaussians.scales,
        densities=gaussians.densities,
        features=gaussians.features,
        n_active_features=gaussians.active_sh_degree,
        gpu_batch=batch,
    )
    return _format_render_pkg(render_pkg, gaussians.positions, bg_color)


def render_rolling_shutter(
    tracer,
    state_at_time,
    camera: CameraState,
    effects: RenderEffects,
    rolling_cfg: RollingShutterConfig,
    bg_color: torch.Tensor | None = None,
    rebuild=True,
):
    if camera.time is None:
        raise ValueError("camera.time is required for rolling-shutter rendering.")

    cur_time = float(camera.time)
    prev_time = max(0.0, cur_time - rolling_cfg.shutter_time / 2.0)
    next_time = min(float(rolling_cfg.maxtime), cur_time + rolling_cfg.shutter_time / 2.0)
    total_time = next_time - prev_time
    im_w = int(camera.width)
    im_h = int(camera.height)

    def row_time(x, y):
        if rolling_cfg.shutter_type == "global":
            return float((prev_time + next_time) / 2.0)
        if rolling_cfg.shutter_type == "rolling_tb":
            return float(float(math.floor(y)) / (im_h - 1) * total_time + prev_time)
        if rolling_cfg.shutter_type == "rolling_bt":
            return float((im_h - float(math.ceil(y) - 1)) / (im_h - 1) * total_time + prev_time)
        if rolling_cfg.shutter_type == "rolling_lr":
            return float(float(math.floor(x)) / (im_w - 1) * total_time + prev_time)
        if rolling_cfg.shutter_type == "rolling_rl":
            return float((im_w - float(math.ceil(x) - 1)) / (im_w - 1) * total_time + prev_time)
        raise ValueError(f"Unknown shutter type: {rolling_cfg.shutter_type}")

    if rolling_cfg.shutter_type == "global":
        state = state_at_time(float((prev_time + next_time) / 2.0))
        return render_gaussian_state(tracer, state, camera, effects, bg_color=bg_color, rebuild=rebuild)

    rgb_buffer = None
    depth_buffer = None
    opacity_buffer = None
    axis = "row" if rolling_cfg.shutter_type in ("rolling_tb", "rolling_bt") else "col"
    if axis == "row":
        indices = [y if rolling_cfg.shutter_type == "rolling_tb" else im_h - y - 1 for y in range(im_h)]
    else:
        indices = [x if rolling_cfg.shutter_type == "rolling_lr" else im_w - x - 1 for x in range(im_w)]

    row_chunk_size = int(rolling_cfg.row_chunk_size)
    rebuild_every = int(rolling_cfg.rebuild_every)
    if row_chunk_size <= 0 or rebuild_every <= 0:
        raise ValueError("row_chunk_size and rebuild_every must be positive.")

    for idx in range(0, len(indices), row_chunk_size):
        chunk_indices = indices[idx : idx + row_chunk_size]
        if len(chunk_indices) != row_chunk_size:
            continue
        if row_chunk_size > 1:
            times = [
                row_time(chunk_idx if axis == "col" else 0, chunk_idx if axis == "row" else 0)
                for chunk_idx in chunk_indices
            ]
            time_value = sum(times) / row_chunk_size
        else:
            time_value = row_time(idx if axis == "col" else 0, idx if axis == "row" else 0)
        time_value = max(prev_time, min(float(time_value), next_time))
        state = state_at_time(time_value).validate()
        if rgb_buffer is None:
            rgb_buffer = torch.zeros((im_h, im_w, 3), dtype=torch.float32, device=state.positions.device)
            depth_buffer = torch.zeros((im_h, im_w, 1), dtype=torch.float32, device=state.positions.device)
            opacity_buffer = torch.zeros((im_h, im_w, 1), dtype=torch.float32, device=state.positions.device)

        batch = camera_state_to_batch(camera, effects, device=state.positions.device)
        row_mask = torch.zeros((im_h, im_w), dtype=torch.bool, device=state.positions.device)
        for chunk_idx in chunk_indices:
            if axis == "row":
                row_mask[chunk_idx, :] = True
            else:
                row_mask[:, chunk_idx] = True
        batch = mask_batch(batch, row_mask, make_square=True)

        tracer.build_acc_custom(
            positions=state.positions,
            rotations=state.rotations,
            scales=state.scales,
            densities=state.densities,
            rebuild=(idx % rebuild_every == 0) or rebuild,
        )
        row_pkg = tracer.render_custom(
            positions=state.positions,
            rotations=state.rotations,
            scales=state.scales,
            densities=state.densities,
            features=state.features,
            n_active_features=state.active_sh_degree,
            gpu_batch=batch,
        )
        batch_size = row_pkg["pred_rgb"].shape[0]
        row_rgb = row_pkg["pred_rgb"].reshape(batch_size, row_chunk_size, -1, 3).mean(0)
        row_depth = row_pkg["pred_dist"].reshape(batch_size, row_chunk_size, -1, 1).mean(0)
        row_opacity = row_pkg["pred_opacity"].reshape(batch_size, row_chunk_size, -1, 1).mean(0)
        start = idx
        end = idx + row_chunk_size
        if axis == "row":
            rgb_buffer[start:end, :] = row_rgb
            depth_buffer[start:end, :] = row_depth
            opacity_buffer[start:end, :] = row_opacity
        else:
            rgb_buffer[:, start:end] = row_rgb
            depth_buffer[:, start:end] = row_depth
            opacity_buffer[:, start:end] = row_opacity

    pred_rgb = rgb_buffer.permute(2, 0, 1)
    pred_depth = depth_buffer.permute(2, 0, 1)
    pred_opacity = opacity_buffer.permute(2, 0, 1)
    if bg_color is not None:
        pred_rgb = _composite_background(pred_rgb, pred_opacity, bg_color)
    return {
        "render": pred_rgb,
        "depth": pred_depth,
        "opacity": pred_opacity,
        "means3Dfinal": None,
    }


def _format_render_pkg(render_pkg, means3d_final, bg_color):
    pred_rgb = render_pkg["pred_rgb"].mean(0).permute(2, 0, 1)
    pred_depth = render_pkg["pred_dist"].mean(0).permute(2, 0, 1)
    pred_opacity = render_pkg["pred_opacity"].mean(0).permute(2, 0, 1)
    if bg_color is not None:
        pred_rgb = _composite_background(pred_rgb, pred_opacity, bg_color)
    return {
        "render": pred_rgb,
        "viewspace_points": None,
        "visibility_filter": None,
        "radii": None,
        "depth": pred_depth,
        "opacity": pred_opacity,
        "means3Dfinal": means3d_final,
    }


def _composite_background(pred_rgb, pred_opacity, bg_color):
    bg_color_expanded = bg_color.unsqueeze(1).unsqueeze(2)
    bg_color_expanded = bg_color_expanded.expand(-1, pred_rgb.shape[1], pred_rgb.shape[2])
    return pred_rgb + bg_color_expanded * (1.0 - pred_opacity)


class GrutRenderer:
    def __init__(self, mode=None, config=None, tracer=None):
        self.config = config
        self.mode = mode or (None if config is None else config.render.method)
        self.tracer = tracer if tracer is not None else create_tracer(self.mode, config)

    def render(self, gaussians: GaussianState, camera: CameraState, effects: RenderEffects | None = None, bg_color=None, rebuild=True):
        return render_gaussian_state(
            self.tracer,
            gaussians,
            camera,
            effects=effects,
            bg_color=bg_color,
            rebuild=rebuild,
        )
