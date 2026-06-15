from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class DofConfig:
    spp: int
    dof: object


@dataclass
class RollingShutterConfig:
    shutter_type: str = "global"
    shutter_time: float = 0.05
    maxtime: float = 1.0
    row_chunk_size: int = 1
    rebuild_every: int = 128


@dataclass
class RenderEffects:
    trace_camera_type: str = "pinhole"
    radial_coeffs: Optional[Sequence[float]] = None
    fisheye_fov_deg: float = 180.0
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 36.0
    dof: Optional[object] = None
    shutter_type: str = "global"
    rolling_shutter: Optional[RollingShutterConfig] = None
