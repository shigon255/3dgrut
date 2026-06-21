from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CameraState:
    """Camera and optional supervision data required by 3DGRUT."""

    width: int
    height: int
    fovx: float
    fovy: float
    c2w: torch.Tensor
    image: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    time: Optional[float] = None
    last_c2w: Optional[torch.Tensor] = None
    last_time: Optional[float] = None

    def validate(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Camera resolution must be positive, got {self.width}x{self.height}.")
        if self.fovx <= 0 or self.fovy <= 0:
            raise ValueError(f"Camera FoV must be positive, got {self.fovx}, {self.fovy}.")
        if tuple(self.c2w.shape[-2:]) != (4, 4):
            raise ValueError(f"c2w must end with shape [4, 4], got {self.c2w.shape}.")
        return self
