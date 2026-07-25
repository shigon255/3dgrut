from types import SimpleNamespace

import pytest
import torch

from threedgrut_gaussian_adapter import (
    CameraState,
    GaussianState,
    RenderEffects,
    RollingShutterConfig,
)
from threedgrut_gaussian_adapter import renderer as rolling_renderer


def _state():
    return GaussianState(
        positions=torch.zeros((1, 3)),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.ones((1, 3)),
        densities=torch.ones((1, 1)),
        features=torch.zeros((1, 3)),
        active_sh_degree=0,
    ).validate()


def _camera(width=5, height=2):
    pose = torch.eye(4)
    return CameraState(
        width=width, height=height, fovx=1.0, fovy=1.0, c2w=pose,
        time=1.0, last_c2w=pose.clone(), last_time=0.0,
    ).validate()


def test_rolling_flat_index_chunks_keeps_a_partial_final_column_chunk():
    chunks = rolling_renderer._rolling_flat_index_chunks(
        "col", list(range(5)), row_chunk_size=3, im_w=5, im_h=2, device="cpu"
    )

    assert len(chunks) == 2
    assert torch.equal(chunks[0], torch.tensor([0, 1, 2, 5, 6, 7]))
    assert torch.equal(chunks[1], torch.tensor([3, 4, 8, 9]))


def test_rolling_flat_index_chunks_keeps_a_partial_final_row_chunk():
    chunks = rolling_renderer._rolling_flat_index_chunks(
        "row", list(range(5)), row_chunk_size=3, im_w=2, im_h=5, device="cpu"
    )

    assert len(chunks) == 2
    assert torch.equal(chunks[0], torch.tensor([0, 1, 2, 3, 4, 5]))
    assert torch.equal(chunks[1], torch.tensor([6, 7, 8, 9]))


def test_rolling_shutter_places_and_covers_a_partial_final_column_chunk(monkeypatch):
    calls = []

    def fake_camera_state_to_batch(*args, **kwargs):
        return SimpleNamespace()

    def fake_slice_batch_flat_indices(*, flat_indices, **kwargs):
        return SimpleNamespace(flat_indices=flat_indices)

    class FakeTracer:
        def build_acc_custom(self, **kwargs):
            return None

        def render_custom(self, *, gpu_batch, **kwargs):
            calls.append(gpu_batch.flat_indices)
            pixels = gpu_batch.flat_indices.numel()
            value = float(len(calls))
            return {
                "pred_rgb": torch.full((1, pixels, 3), value),
                "pred_dist": torch.full((1, pixels, 1), value),
                "pred_opacity": torch.ones((1, pixels, 1)),
            }

    monkeypatch.setattr(rolling_renderer, "camera_state_to_batch", fake_camera_state_to_batch)
    monkeypatch.setattr(rolling_renderer, "_batch_to_world_space", lambda batch: batch)
    monkeypatch.setattr(rolling_renderer, "slice_batch_flat_indices", fake_slice_batch_flat_indices)
    sampled_times = []

    def state_at_time(value):
        sampled_times.append(value)
        return _state()

    output = rolling_renderer.render_rolling_shutter(
        FakeTracer(), state_at_time, _camera(), RenderEffects(),
        RollingShutterConfig(shutter_type="rolling_lr", row_chunk_size=3, rebuild_every=1),
    )

    assert len(calls) == 2
    assert sampled_times == pytest.approx([0.25, 0.875])
    assert output["render"].shape == (3, 2, 5)
    assert torch.equal(output["render"][0, :, :3], torch.ones((2, 3)))
    assert torch.equal(output["render"][0, :, 3:], torch.full((2, 2), 2.0))
    assert torch.all(output["opacity"] > 0)


@pytest.mark.parametrize(
    ("shutter_type", "width", "height"),
    [
        ("rolling_lr", 1, 2),
        ("rolling_rl", 1, 2),
        ("rolling_tb", 2, 1),
        ("rolling_bt", 2, 1),
    ],
)
def test_rolling_shutter_uses_midpoint_for_a_unit_scan_axis(
    monkeypatch, shutter_type, width, height
):
    monkeypatch.setattr(
        rolling_renderer, "camera_state_to_batch", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(rolling_renderer, "_batch_to_world_space", lambda batch: batch)
    monkeypatch.setattr(
        rolling_renderer,
        "slice_batch_flat_indices",
        lambda *, flat_indices, **kwargs: SimpleNamespace(flat_indices=flat_indices),
    )

    class FakeTracer:
        def build_acc_custom(self, **kwargs):
            return None

        def render_custom(self, *, gpu_batch, **kwargs):
            pixels = gpu_batch.flat_indices.numel()
            return {
                "pred_rgb": torch.ones((1, pixels, 3)),
                "pred_dist": torch.ones((1, pixels, 1)),
                "pred_opacity": torch.ones((1, pixels, 1)),
            }

    sampled_times = []

    def state_at_time(value):
        sampled_times.append(value)
        return _state()

    output = rolling_renderer.render_rolling_shutter(
        FakeTracer(), state_at_time, _camera(width, height), RenderEffects(),
        RollingShutterConfig(shutter_type=shutter_type, row_chunk_size=1, rebuild_every=1),
    )

    assert sampled_times == pytest.approx([0.5])
    assert torch.all(output["render"] == 1)
    assert torch.all(output["opacity"] == 1)
