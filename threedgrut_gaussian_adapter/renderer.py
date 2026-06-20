import math
import time

import torch

from .camera import CameraState
from .camera_models import camera_state_to_batch, slice_batch_flat_indices
from .effects import RenderEffects, RollingShutterConfig
from .gaussian_state import GaussianState


def _profile_now(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _format_profile(prefix, timings):
    parts = [f"{key}={value * 1000.0:.2f}ms" for key, value in timings.items()]
    print(f"[render-profile] {prefix}: " + ", ".join(parts), flush=True)


def _rolling_flat_indices(axis, chunk_indices, im_w, im_h, device):
    chunk = torch.as_tensor(chunk_indices, dtype=torch.long, device=device)
    if axis == "row":
        flat = chunk[:, None] * im_w + torch.arange(im_w, dtype=torch.long, device=device)[None, :]
    else:
        flat = torch.arange(im_h, dtype=torch.long, device=device)[:, None] * im_w + chunk[None, :]
    # Boolean masking returned pixels in ascending flat image order; preserve that ordering.
    return torch.sort(flat.reshape(-1)).values


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
    profile = bool(getattr(effects, "profile", False))
    device = gaussians.positions.device
    timings = {}
    start = _profile_now(device) if profile else None

    batch = camera_state_to_batch(
        camera,
        effects,
        device=device,
        use_rolling_shutter=False,
    )
    if profile:
        now = _profile_now(device)
        timings["batch"] = now - start
        start = now

    tracer.build_acc_custom(
        positions=gaussians.positions,
        rotations=gaussians.rotations,
        scales=gaussians.scales,
        densities=gaussians.densities,
        rebuild=rebuild,
    )
    if profile:
        now = _profile_now(device)
        timings["build"] = now - start
        start = now

    render_pkg = tracer.render_custom(
        positions=gaussians.positions,
        rotations=gaussians.rotations,
        scales=gaussians.scales,
        densities=gaussians.densities,
        features=gaussians.features,
        n_active_features=gaussians.active_sh_degree,
        gpu_batch=batch,
    )
    if profile:
        now = _profile_now(device)
        timings["trace"] = now - start
        start = now

    result = _format_render_pkg(render_pkg, gaussians.positions, bg_color)
    if profile:
        timings["format"] = _profile_now(device) - start
        _format_profile("global", timings)
    return result


def render_rolling_shutter(
    tracer,
    state_at_time,
    camera: CameraState,
    effects: RenderEffects,
    rolling_cfg: RollingShutterConfig,
    bg_color: torch.Tensor | None = None,
    rebuild=True,
    states_at_times=None,
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
    base_batch = None
    flat_index_chunks = None
    timings = {"state": 0.0, "batch": 0.0, "build": 0.0, "trace": 0.0}
    profile = bool(getattr(effects, "profile", False))
    axis = "row" if rolling_cfg.shutter_type in ("rolling_tb", "rolling_bt") else "col"
    if axis == "row":
        indices = [y if rolling_cfg.shutter_type == "rolling_tb" else im_h - y - 1 for y in range(im_h)]
    else:
        indices = [x if rolling_cfg.shutter_type == "rolling_lr" else im_w - x - 1 for x in range(im_w)]

    row_chunk_size = int(rolling_cfg.row_chunk_size)
    rebuild_every = int(rolling_cfg.rebuild_every)
    state_batch_size = int(getattr(rolling_cfg, "state_batch_size", 1))
    if row_chunk_size <= 0 or rebuild_every <= 0 or state_batch_size <= 0:
        raise ValueError("row_chunk_size, rebuild_every, and state_batch_size must be positive.")

    total_chunks = len(indices) // row_chunk_size
    progress_interval = max(1, total_chunks // 20)
    print(
        f"[rolling-shutter] start image: shutter={rolling_cfg.shutter_type}, "
        f"axis={axis}, resolution={im_w}x{im_h}, chunks={total_chunks}, "
        f"chunk_size={row_chunk_size}, rebuild_every={rebuild_every}, "
        f"state_batch_size={state_batch_size}",
        flush=True,
    )

    state_queue = {}

    def chunk_time(offset, chunk_indices):
        if row_chunk_size > 1:
            times = [
                row_time(chunk_idx if axis == "col" else 0, chunk_idx if axis == "row" else 0)
                for chunk_idx in chunk_indices
            ]
            value = sum(times) / row_chunk_size
        else:
            value = row_time(offset if axis == "col" else 0, offset if axis == "row" else 0)
        return max(prev_time, min(float(value), next_time))

    for idx in range(0, len(indices), row_chunk_size):
        chunk_indices = indices[idx : idx + row_chunk_size]
        if len(chunk_indices) != row_chunk_size:
            continue
        chunk_number = idx // row_chunk_size + 1
        if chunk_number == 1 or chunk_number == total_chunks or chunk_number % progress_interval == 0:
            print(
                f"[rolling-shutter] {axis} chunk {chunk_number}/{total_chunks} "
                f"({100.0 * chunk_number / total_chunks:.1f}%)",
                flush=True,
            )
        time_value = chunk_time(idx, chunk_indices)
        start = _profile_now("cuda") if profile and torch.cuda.is_available() else (time.perf_counter() if profile else None)
        if states_at_times is not None and state_batch_size > 1:
            if chunk_number not in state_queue:
                group_offsets = list(range(idx, min(len(indices), idx + row_chunk_size * state_batch_size), row_chunk_size))
                group_records = []
                group_times = []
                for group_offset in group_offsets:
                    group_indices = indices[group_offset : group_offset + row_chunk_size]
                    if len(group_indices) != row_chunk_size:
                        continue
                    group_number = group_offset // row_chunk_size + 1
                    group_records.append(group_number)
                    group_times.append(chunk_time(group_offset, group_indices))
                for group_number, group_state in zip(group_records, states_at_times(group_times)):
                    state_queue[group_number] = group_state
            state = state_queue.pop(chunk_number).validate()
        else:
            state = state_at_time(time_value).validate()
        if profile:
            now = _profile_now(state.positions.device)
            timings["state"] += now - start
            start = now
        if rgb_buffer is None:
            rgb_buffer = torch.zeros((im_h, im_w, 3), dtype=torch.float32, device=state.positions.device)
            depth_buffer = torch.zeros((im_h, im_w, 1), dtype=torch.float32, device=state.positions.device)
            opacity_buffer = torch.zeros((im_h, im_w, 1), dtype=torch.float32, device=state.positions.device)
            base_batch = camera_state_to_batch(camera, effects, device=state.positions.device)
            flat_index_chunks = [
                _rolling_flat_indices(axis, indices[offset : offset + row_chunk_size], im_w, im_h, state.positions.device)
                for offset in range(0, len(indices), row_chunk_size)
                if len(indices[offset : offset + row_chunk_size]) == row_chunk_size
            ]

        batch = slice_batch_flat_indices(batch=base_batch, flat_indices=flat_index_chunks[chunk_number - 1], make_square=True)
        if profile:
            now = _profile_now(state.positions.device)
            timings["batch"] += now - start
            start = now

        tracer.build_acc_custom(
            positions=state.positions,
            rotations=state.rotations,
            scales=state.scales,
            densities=state.densities,
            rebuild=(idx % rebuild_every == 0) or rebuild,
        )
        if profile:
            now = _profile_now(state.positions.device)
            timings["build"] += now - start
            start = now
        row_pkg = tracer.render_custom(
            positions=state.positions,
            rotations=state.rotations,
            scales=state.scales,
            densities=state.densities,
            features=state.features,
            n_active_features=state.active_sh_degree,
            gpu_batch=batch,
        )
        if profile:
            timings["trace"] += _profile_now(state.positions.device) - start
        batch_size = row_pkg["pred_rgb"].shape[0]
        expected_pixels = row_chunk_size * (im_w if axis == "row" else im_h)

        def crop_padded_chunk(tensor, channels):
            flattened = tensor.reshape(batch_size, -1, channels)
            if flattened.shape[1] < expected_pixels:
                raise RuntimeError(
                    f"Rolling-shutter render returned {flattened.shape[1]} pixels, "
                    f"expected at least {expected_pixels}."
                )
            averaged = flattened[:, :expected_pixels, :].mean(0)
            if axis == "row":
                return averaged.reshape(row_chunk_size, im_w, channels)
            return averaged.reshape(im_h, row_chunk_size, channels)

        row_rgb = crop_padded_chunk(row_pkg["pred_rgb"], 3)
        row_depth = crop_padded_chunk(row_pkg["pred_dist"], 1)
        row_opacity = crop_padded_chunk(row_pkg["pred_opacity"], 1)
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

    print("[rolling-shutter] finished image", flush=True)
    if profile:
        _format_profile("rolling-shutter", timings)

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
