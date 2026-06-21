import numpy as np
import torch

from threedgrut.datasets.camera_models import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
    pixels_to_image_points,
)
try:
    from threedgrut.datasets.camera_models import BlenderFisheyeCameraModelParameters
except ImportError:
    BlenderFisheyeCameraModelParameters = None
from threedgrut.datasets.protocols import Batch, TwoBatch
from threedgrut.datasets.utils import compute_max_radius
try:
    from threedgrut_playground.utils.rng import rng_torch_low_discrepancy
except ImportError:
    rng_torch_low_discrepancy = None

try:
    from threedgrut_playground.utils.depth_of_field import RayBundle
except ImportError:
    class RayBundle:
        def __init__(self, rays_o, rays_d, pixel_x, pixel_y):
            self.rays_ori = rays_o
            self.rays_dir = rays_d
            self.pixel_x = pixel_x
            self.pixel_y = pixel_y


from .camera import CameraState
from .effects import RenderEffects


SHUTTER_TYPE_MAP = {
    "global": ShutterType.GLOBAL,
    "rolling_tb": ShutterType.ROLLING_TOP_TO_BOTTOM,
    "rolling_bt": ShutterType.ROLLING_BOTTOM_TO_TOP,
    "rolling_lr": ShutterType.ROLLING_LEFT_TO_RIGHT,
    "rolling_rl": ShutterType.ROLLING_RIGHT_TO_LEFT,
}

_CAMERA_TEMPLATE_CACHE = {}
_MAX_CAMERA_TEMPLATE_CACHE_SIZE = 16


def fov2focal(fov, pixels):
    return pixels / (2.0 * np.tan(float(fov) * 0.5))


def pinhole_camera_rays(x, y, f_x, f_y, w, h, ray_jitter=None):
    if ray_jitter is not None:
        jitter = ray_jitter(x.shape).numpy()
        jitter_xs = jitter[:, 0]
        jitter_ys = jitter[:, 1]
    else:
        jitter_xs = jitter_ys = 0.5

    xs = ((x + jitter_xs) - 0.5 * w) / f_x
    ys = ((y + jitter_ys) - 0.5 * h) / f_y
    ray_lookat = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    ray_origin = np.zeros_like(ray_lookat)
    return ray_origin, ray_lookat / np.linalg.norm(ray_lookat, axis=-1, keepdims=True)


def create_pinhole_camera(focalx, focaly, w, h, ray_jitter=None, shutter_type=ShutterType.GLOBAL):
    u = np.tile(np.arange(w), h)
    v = np.arange(h).repeat(w)
    out_shape = (1, h, w, 3)
    params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([w, h], dtype=np.int64),
        shutter_type=shutter_type,
        principal_point=np.array([w, h], dtype=np.float32) / 2,
        focal_length=np.array([focalx, focaly], dtype=np.float32),
        radial_coeffs=np.zeros((6,), dtype=np.float32),
        tangential_coeffs=np.zeros((2,), dtype=np.float32),
        thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
    )
    rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, focalx, focaly, w, h, ray_jitter)
    return (
        params.to_dict(),
        torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
        torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
        type(params).__name__,
        torch.tensor(u, dtype=torch.int32).reshape(h, w),
        torch.tensor(v, dtype=torch.int32).reshape(h, w),
    )


def image_points_to_camera_rays_blender_mm(camera_model_parameters, image_points, sensor_size_mm, device="cpu"):
    dtype = torch.float32
    pp = torch.tensor(camera_model_parameters.principal_point, dtype=dtype, device=device)
    res = torch.tensor(camera_model_parameters.resolution.astype(np.int32), device=device)
    coeffs = torch.tensor(camera_model_parameters.radial_coeffs, dtype=dtype, device=device)
    if coeffs.numel() == 4:
        coeffs = torch.cat([torch.zeros(1, dtype=dtype, device=device), coeffs])

    max_angle = float(getattr(camera_model_parameters, "max_angle", np.pi))
    pix = image_points.to(dtype) - pp
    x_mm = pix[:, 0:1] * (float(sensor_size_mm[0]) / float(res[0]))
    y_mm = pix[:, 1:2] * (float(sensor_size_mm[1]) / float(res[1]))
    r_mm = torch.sqrt(x_mm**2 + y_mm**2 + 1e-9)
    coeffs = coeffs * (np.pi / 180.0)
    thetas = coeffs[0] + coeffs[1] * r_mm + coeffs[2] * r_mm**2 + coeffs[3] * r_mm**3 + coeffs[4] * r_mm**4
    thetas = torch.clamp(thetas, max=max_angle)
    phi = torch.atan2(y_mm, x_mm)
    return torch.cat(
        [
            torch.sin(thetas) * torch.cos(phi),
            torch.sin(thetas) * torch.sin(phi),
            torch.cos(thetas),
        ],
        1,
    )


def create_fisheye_camera(
    focalx,
    focaly,
    w,
    h,
    radial_coeffs=None,
    fisheye_fov=180.0,
    sensor_width_mm=36.0,
    sensor_height_mm=36.0,
    shutter_type=ShutterType.GLOBAL,
):
    radial_coeffs = [0.0, 1.0, 0.03, 0.001, 0.0] if radial_coeffs is None else list(radial_coeffs)
    u = np.tile(np.arange(w), h)
    v = np.arange(h).repeat(w)
    out_shape = (1, h, w, 3)
    params_array = np.array([focalx, focaly, w / 2, h / 2] + radial_coeffs).astype(np.float32)
    resolution = np.array([w, h]).astype(np.int64)
    principal_point = params_array[2:4].astype(np.float32)
    focal_length = params_array[0:2].astype(np.float32)
    coeffs = params_array[4:].astype(np.float32)
    if fisheye_fov is None:
        max_radius_pixels = compute_max_radius(resolution.astype(np.float64), principal_point)
        fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
        fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
        max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0
    else:
        max_angle = float(fisheye_fov) * (np.pi / 180.0) / 2.0
    params = OpenCVFisheyeCameraModelParameters(
        principal_point=principal_point,
        focal_length=focal_length,
        radial_coeffs=coeffs,
        resolution=resolution,
        max_angle=max_angle,
        shutter_type=shutter_type,
    )
    pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
    image_points = pixels_to_image_points(pixel_coords)
    rays_d_cam = image_points_to_camera_rays_blender_mm(
        params, image_points, sensor_size_mm=(sensor_width_mm, sensor_height_mm), device="cpu"
    )
    rays_o_cam = torch.zeros_like(rays_d_cam)
    return (
        params.to_dict(),
        rays_o_cam.to(torch.float32).reshape(out_shape),
        rays_d_cam.to(torch.float32).reshape(out_shape),
        type(params).__name__,
        torch.tensor(u, dtype=torch.int32).reshape(h, w),
        torch.tensor(v, dtype=torch.int32).reshape(h, w),
    )


def fisheye_to_direction_bl(theta, x_mm, y_mm, r, eps=1e-9):
    cos_phi = torch.clamp(x_mm / (r + eps), -1.0, 1.0)
    phi = torch.acos(cos_phi)
    phi = torch.where(y_mm < 0, -phi, phi)
    return torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )


def generate_lens_polynomial_rays_bl(
    width,
    height,
    blender_coeffs=None,
    sensor_width_mm=36.0,
    sensor_height_mm=36.0,
    fisheye_fov_deg=180.0,
    device="cpu",
    eps=1e-9,
):
    blender_coeffs = (
        [0.0, 0.017453292, 0.00052359875, 1.7453303e-5, 0.0]
        if blender_coeffs is None
        else list(blender_coeffs)
    )
    px = torch.arange(width, device=device)
    py = torch.arange(height, device=device)
    grid_x, grid_y = torch.meshgrid(px, py, indexing="xy")
    grid_x = grid_x.float()
    grid_y = grid_y.float()
    center_x, center_y = width / 2.0, height / 2.0
    if width >= height:
        sensor_height_mm = sensor_width_mm * (height / width)
    else:
        sensor_width_mm = sensor_height_mm * (width / height)
    x_mm = (grid_x - center_x) / width * sensor_width_mm
    y_mm = (grid_y - center_y) / height * sensor_height_mm
    r = torch.sqrt(x_mm**2 + y_mm**2 + eps)
    k0, k1, k2, k3, k4 = blender_coeffs[:5]
    r2 = r**2
    theta = k0 + k1 * r + k2 * r2 + k3 * r * r2 + k4 * r2 * r2
    fov = torch.deg2rad(torch.tensor(fisheye_fov_deg, dtype=torch.float32, device=device))
    theta = torch.where(torch.abs(theta) > 0.5 * fov, torch.tensor(0.0, device=device), theta)
    return fisheye_to_direction_bl(theta, x_mm, y_mm, r, eps=eps)


def create_blender_fisheye_camera(
    w,
    h,
    blender_coeffs=None,
    sensor_width_mm=36.0,
    sensor_height_mm=36.0,
    fisheye_fov_deg=180.0,
    shutter_type=ShutterType.GLOBAL,
):
    blender_coeffs = (
        [0.0, 0.017453292, 0.00052359875, 1.7453303e-5, 0.0]
        if blender_coeffs is None
        else list(blender_coeffs)
    )
    rays_d_cam = generate_lens_polynomial_rays_bl(
        width=w,
        height=h,
        blender_coeffs=blender_coeffs,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        fisheye_fov_deg=fisheye_fov_deg,
        device="cpu",
    )
    rays_o_cam = torch.zeros_like(rays_d_cam)
    out_shape = (1, h, w, 3)
    u = np.tile(np.arange(w), h)
    v = np.arange(h).repeat(w)
    if BlenderFisheyeCameraModelParameters is not None:
        params = BlenderFisheyeCameraModelParameters(
            resolution=np.array([w, h]).astype(np.int64),
            shutter_type=shutter_type,
            blender_coeffs=np.array(blender_coeffs, dtype=np.float32),
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            fisheye_fov_deg=fisheye_fov_deg,
        )
        params_dict = params.to_dict()
    else:
        params_dict = {
            "resolution": np.array([w, h]).astype(np.int64),
            "shutter_type": shutter_type,
            "blender_coeffs": np.array(blender_coeffs, dtype=np.float32),
            "sensor_width_mm": float(sensor_width_mm),
            "sensor_height_mm": float(sensor_height_mm),
            "fisheye_fov_deg": float(fisheye_fov_deg),
        }
    params_dict.update({
        "principal_point": np.array([w, h], dtype=np.float32) / 2,
        "focal_length": np.array([fov2focal(np.deg2rad(fisheye_fov_deg), w), fov2focal(np.deg2rad(fisheye_fov_deg), h)], dtype=np.float32),
        "radial_coeffs": np.array(blender_coeffs, dtype=np.float32),
        "max_angle": float(fisheye_fov_deg) * np.pi / 180.0 / 2.0,
    })
    return (
        params_dict,
        rays_o_cam.to(torch.float32).reshape(out_shape),
        rays_d_cam.to(torch.float32).reshape(out_shape),
        "BlenderFisheyeCameraModelParameters",
        torch.tensor(u, dtype=torch.int32).reshape(h, w),
        torch.tensor(v, dtype=torch.int32).reshape(h, w),
    )


def _normalize_device(device):
    return str(torch.device(device))


def _float_key(value):
    return None if value is None else float(value)


def _sequence_key(values):
    if values is None:
        return None
    return tuple(float(v) for v in values)


def _camera_template_key(camera: CameraState, effects: RenderEffects, device):
    return (
        int(camera.width),
        int(camera.height),
        float(camera.fovx),
        float(camera.fovy),
        effects.trace_camera_type,
        _sequence_key(effects.radial_coeffs),
        _float_key(effects.fisheye_fov_deg),
        _float_key(effects.sensor_width_mm),
        _float_key(effects.sensor_height_mm),
        effects.shutter_type,
        _normalize_device(device),
    )


def _cache_put(key, value):
    if len(_CAMERA_TEMPLATE_CACHE) >= _MAX_CAMERA_TEMPLATE_CACHE_SIZE:
        _CAMERA_TEMPLATE_CACHE.pop(next(iter(_CAMERA_TEMPLATE_CACHE)))
    _CAMERA_TEMPLATE_CACHE[key] = value


def get_camera_template(camera: CameraState, effects: RenderEffects, device="cuda:0", use_cache=True):
    camera.validate()
    key = _camera_template_key(camera, effects, device)
    if use_cache and key in _CAMERA_TEMPLATE_CACHE:
        return _CAMERA_TEMPLATE_CACHE[key]

    focalx = fov2focal(camera.fovx, camera.width)
    focaly = fov2focal(camera.fovy, camera.height)
    shutter_type = SHUTTER_TYPE_MAP.get(effects.shutter_type, ShutterType.GLOBAL)
    radial_coeffs = effects.radial_coeffs

    if effects.trace_camera_type == "pinhole":
        template = create_pinhole_camera(
            focalx, focaly, camera.width, camera.height, shutter_type=shutter_type
        )
    elif effects.trace_camera_type == "fisheye":
        template = create_fisheye_camera(
            focalx,
            focaly,
            camera.width,
            camera.height,
            radial_coeffs=radial_coeffs,
            fisheye_fov=effects.fisheye_fov_deg,
            sensor_width_mm=effects.sensor_width_mm,
            sensor_height_mm=effects.sensor_height_mm,
            shutter_type=shutter_type,
        )
    elif effects.trace_camera_type == "blender_fisheye":
        template = create_blender_fisheye_camera(
            camera.width,
            camera.height,
            blender_coeffs=radial_coeffs,
            sensor_width_mm=effects.sensor_width_mm,
            sensor_height_mm=effects.sensor_height_mm,
            fisheye_fov_deg=effects.fisheye_fov_deg,
            shutter_type=shutter_type,
        )
    else:
        raise ValueError(f"Unsupported trace camera type: {effects.trace_camera_type}")

    cam_param_dict, rays_o, rays_d, cam_param_name, pixel_x, pixel_y = template
    cached_template = (
        cam_param_dict,
        rays_o.to(device=device, non_blocking=True),
        rays_d.to(device=device, non_blocking=True),
        cam_param_name,
        pixel_x.to(device=device, non_blocking=True),
        pixel_y.to(device=device, non_blocking=True),
    )
    if use_cache:
        _cache_put(key, cached_template)
    return cached_template


def _pixel_to_disc_shirley(seed):
    a = 2.0 * seed[:, 0] - 1.0
    b = 2.0 * seed[:, 1] - 1.0
    mask = a * a > b * b
    pi = torch.pi
    r = torch.where(mask, a, b)
    phi = torch.where(mask, (pi / 4.0) * (b / a), (pi / 4.0) * (a / b) + (pi / 2.0))
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi)))


def _dof_inner(dof):
    return getattr(dof, "dof", dof)


def _generate_dof_rays_vectorized(camera_R, rays, dof, sample_start=1, sample_count=None):
    if rng_torch_low_discrepancy is None:
        return None
    depth_of_field = _dof_inner(dof)
    if getattr(depth_of_field, "RNG_MODE", None) != "low_discrepancy_seq":
        return None

    h, w = rays.pixel_x.shape
    total_spp = int(dof.get_spp() if hasattr(dof, "get_spp") else depth_of_field.spp)
    spp = total_spp if sample_count is None else int(sample_count)
    ray_count = h * w
    focus_z = float(getattr(dof, "current_focus_z", getattr(depth_of_field, "focus_z")))
    aperture_size = float(getattr(dof, "current_aperture_size", getattr(depth_of_field, "aperture_size")))

    rays_ori = rays.rays_ori
    rays_dir = rays.rays_dir
    device = rays_ori.device
    camera_R = camera_R.to(device=device, dtype=torch.float32)
    base_seed = (rays.pixel_x.long() * 19349663 + rays.pixel_y.long() * 96925573).reshape(ray_count) & 0xFFFFFFFF
    sample_index = torch.arange(sample_start, sample_start + spp, device=device, dtype=torch.long).reshape(spp, 1).expand(spp, ray_count)
    seed = base_seed.reshape(1, ray_count).expand(spp, ray_count).reshape(-1)
    seed = rng_torch_low_discrepancy(sample_index.reshape(-1), seed)
    seed = torch.stack(seed, dim=1)

    blur = aperture_size * _pixel_to_disc_shirley(seed)
    expanded_cam = camera_R[:3, :2][None].expand(spp * ray_count, 3, 2)
    base_ori = rays_ori.expand(spp, h, w, 3).reshape(spp * ray_count, 3)
    lookat = (rays_ori + rays_dir * focus_z).expand(spp, h, w, 3).reshape(spp * ray_count, 3)
    rays_ori = base_ori + (expanded_cam @ blur.T[:, :, None]).reshape_as(base_ori)
    rays_dir = (lookat - rays_ori) / focus_z

    if hasattr(depth_of_field, "spp_accumulated_for_frame"):
        depth_of_field.spp_accumulated_for_frame = sample_start + spp
    return rays_ori.reshape(spp, h, w, 3), rays_dir.reshape(spp, h, w, 3)


def generate_dof_rays(camera_R, rays, dof, sample_start=1, sample_count=None):
    if dof is None:
        raise ValueError("Depth of field parameters must be provided.")
    h, w = rays.pixel_x.shape
    total_spp = dof.get_spp()
    b = total_spp if sample_count is None else int(sample_count)
    output_shape = (b, h, w, 3)
    dof.reset_accumulation()
    vectorized = _generate_dof_rays_vectorized(camera_R, rays, dof, sample_start=sample_start, sample_count=b)
    if vectorized is not None:
        dof_rays_o, dof_rays_d = vectorized
        if dof_rays_o.shape != output_shape or dof_rays_d.shape != output_shape:
            raise ValueError(f"Unexpected DOF ray shape: {dof_rays_o.shape}, {dof_rays_d.shape}; expected {output_shape}.")
        return dof_rays_o, dof_rays_d

    dof_rays_o = []
    dof_rays_d = []
    while dof.has_more_to_accumulate():
        rays_o, rays_d = dof(camera_R, rays)
        dof_rays_o.append(rays_o)
        dof_rays_d.append(rays_d)
    dof_rays_o = torch.cat(dof_rays_o, dim=0)
    dof_rays_d = torch.cat(dof_rays_d, dim=0)
    if dof_rays_o.shape != output_shape or dof_rays_d.shape != output_shape:
        raise ValueError(f"Unexpected DOF ray shape: {dof_rays_o.shape}, {dof_rays_d.shape}; expected {output_shape}.")
    return dof_rays_o, dof_rays_d


def camera_state_to_batch(camera: CameraState, effects: RenderEffects, device="cuda:0", use_rolling_shutter=False, dof_sample_start=1, dof_sample_count=None):
    cam_param_dict, rays_o, rays_d, cam_param_name, pixel_x, pixel_y = get_camera_template(
        camera, effects, device=device, use_cache=effects.cache_camera_batches
    )

    batch_size = (effects.dof.get_spp() if dof_sample_count is None else int(dof_sample_count)) if effects.dof is not None else 1
    c2w = camera.c2w.to(dtype=torch.float32, device=device).reshape(1, 4, 4).expand(batch_size, 4, 4)
    if effects.dof is not None:
        rays = RayBundle(
            rays_o=rays_o,
            rays_d=rays_d,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
        )
        rays_o, rays_d = generate_dof_rays(
            torch.eye(3, dtype=torch.float32, device=rays_o.device),
            rays,
            effects.dof,
            sample_start=dof_sample_start,
            sample_count=batch_size,
        )

    rgb_gt = torch.zeros((batch_size, camera.height, camera.width, 3), dtype=torch.float32, device=device)
    if camera.image is not None:
        rgb_gt = camera.image.reshape(1, 3, camera.height, camera.width).permute(0, 2, 3, 1)
        rgb_gt = rgb_gt.expand(batch_size, camera.height, camera.width, 3)
    mask = None
    if camera.mask is not None:
        mask = camera.mask.reshape(1, camera.height, camera.width, 1).expand(batch_size, camera.height, camera.width, 1)

    sample = {
        "rays_ori": rays_o,
        "rays_dir": rays_d,
        "T_to_world": c2w,
        "rgb_gt": rgb_gt.to(device=device, non_blocking=True),
        "mask": mask.to(device=device, non_blocking=True) if mask is not None else None,
        f"intrinsics_{cam_param_name}": cam_param_dict,
    }
    if use_rolling_shutter:
        if camera.last_c2w is None:
            raise ValueError("last_c2w must be provided for rolling-shutter TwoBatch.")
        sample["T_to_world_last"] = camera.last_c2w.to(dtype=torch.float32).reshape(1, 4, 4).expand(batch_size, 4, 4).to(device=device)
        return TwoBatch(**sample)
    return Batch(**sample)


def slice_batch_flat_indices(batch: Batch, flat_indices: torch.Tensor, make_square=False) -> Batch:
    b, h, w, _ = batch.rays_ori.shape
    flat_indices = flat_indices.to(device=batch.rays_ori.device, dtype=torch.long)
    rays_ori = batch.rays_ori.reshape(b, h * w, 3).index_select(1, flat_indices)
    rays_dir = batch.rays_dir.reshape(b, h * w, 3).index_select(1, flat_indices)
    if make_square:
        n = rays_ori.shape[1]
        side = int(np.ceil(np.sqrt(n)))
        padded_n = side * side
        if padded_n > n:
            pad = padded_n - n
            rays_ori = torch.cat([rays_ori, torch.zeros((b, pad, 3), dtype=rays_ori.dtype, device=rays_ori.device)], dim=1)
            rays_dir = torch.cat([rays_dir, torch.zeros((b, pad, 3), dtype=rays_dir.dtype, device=rays_dir.device)], dim=1)
        rays_ori = rays_ori.reshape(b, side, side, 3)
        rays_dir = rays_dir.reshape(b, side, side, 3)
    else:
        rays_ori = rays_ori.reshape(b, 1, -1, 3)
        rays_dir = rays_dir.reshape(b, 1, -1, 3)
    return Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=batch.T_to_world,
        rgb_gt=None,
        mask=None,
        intrinsics=batch.intrinsics,
        intrinsics_OpenCVPinholeCameraModelParameters=batch.intrinsics_OpenCVPinholeCameraModelParameters,
        intrinsics_OpenCVFisheyeCameraModelParameters=batch.intrinsics_OpenCVFisheyeCameraModelParameters,
        intrinsics_BlenderFisheyeCameraModelParameters=batch.intrinsics_BlenderFisheyeCameraModelParameters,
    )


def mask_batch(batch: Batch, mask: torch.Tensor, make_square=False) -> Batch:
    b, h, w, _ = batch.rays_ori.shape
    if mask.shape != (h, w):
        raise ValueError(f"Mask shape {mask.shape} does not match batch shape {(h, w)}.")
    flat_mask = mask.reshape(-1)
    rays_ori = batch.rays_ori.reshape(b, h * w, 3)[:, flat_mask]
    rays_dir = batch.rays_dir.reshape(b, h * w, 3)[:, flat_mask]
    if make_square:
        n = rays_ori.shape[1]
        side = int(np.ceil(np.sqrt(n)))
        padded_n = side * side
        if padded_n > n:
            pad = padded_n - n
            rays_ori = torch.cat([rays_ori, torch.zeros((b, pad, 3), dtype=rays_ori.dtype, device=rays_ori.device)], dim=1)
            rays_dir = torch.cat([rays_dir, torch.zeros((b, pad, 3), dtype=rays_dir.dtype, device=rays_dir.device)], dim=1)
        rays_ori = rays_ori.reshape(b, side, side, 3)
        rays_dir = rays_dir.reshape(b, side, side, 3)
    else:
        rays_ori = rays_ori.reshape(b, 1, -1, 3)
        rays_dir = rays_dir.reshape(b, 1, -1, 3)
    return Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=batch.T_to_world,
        rgb_gt=None,
        mask=None,
        intrinsics=batch.intrinsics,
        intrinsics_OpenCVPinholeCameraModelParameters=batch.intrinsics_OpenCVPinholeCameraModelParameters,
        intrinsics_OpenCVFisheyeCameraModelParameters=batch.intrinsics_OpenCVFisheyeCameraModelParameters,
        intrinsics_BlenderFisheyeCameraModelParameters=batch.intrinsics_BlenderFisheyeCameraModelParameters,
    )
