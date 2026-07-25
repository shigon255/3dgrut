import numpy as np
import pytest
import torch

from threedgrut_gaussian_adapter.camera_models import create_fisheye_camera


def test_canonical_fisheye_returns_blender_projection_parameters_and_degree_polynomial_rays():
    params, rays_o, rays_d, name, _, _ = create_fisheye_camera(
        focalx=10.0, focaly=10.0, w=4, h=4,
        radial_coeffs=[0.0, 1.0, 0.0, 0.0, 0.0],
        fisheye_fov=180.0, sensor_width_mm=4.0, sensor_height_mm=4.0,
    )
    assert name == "BlenderFisheyeCameraModelParameters"
    assert np.allclose(params["blender_coeffs"], [0.0, np.pi / 180.0, 0.0, 0.0, 0.0])
    assert torch.equal(rays_o, torch.zeros_like(rays_o))
    theta = np.deg2rad(np.sqrt(4.5))
    expected = torch.tensor([-np.sin(theta) / np.sqrt(2.0),
                             -np.sin(theta) / np.sqrt(2.0), np.cos(theta)], dtype=torch.float32)
    assert torch.allclose(rays_d[0, 0, 0], expected, atol=1e-6)

def test_canonical_fisheye_without_explicit_fov_derives_blender_fov_from_max_angle():
    params, rays_o, rays_d, name, _, _ = create_fisheye_camera(
        focalx=10.0, focaly=10.0, w=4, h=4,
        radial_coeffs=[0.0, 1.0, 0.0, 0.0, 0.0],
        fisheye_fov=None, sensor_width_mm=4.0, sensor_height_mm=4.0,
    )
    expected_fov_deg = np.rad2deg(2.0 * (np.sqrt(8.0) / 10.0))
    assert name == "BlenderFisheyeCameraModelParameters"
    assert np.allclose(params["blender_coeffs"], [0.0, np.pi / 180.0, 0.0, 0.0, 0.0])
    assert params["fisheye_fov_deg"] == pytest.approx(expected_fov_deg)
    assert torch.equal(rays_o, torch.zeros_like(rays_o))
    assert torch.isfinite(rays_d).all()
