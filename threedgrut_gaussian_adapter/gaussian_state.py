from dataclasses import dataclass

import torch


@dataclass
class GaussianState:
    """Activated Gaussian tensors ready for 3DGRUT rendering."""

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    features: torch.Tensor
    active_sh_degree: int

    def validate(self):
        n_points = self.positions.shape[0]
        for name, value in (
            ("rotations", self.rotations),
            ("scales", self.scales),
            ("densities", self.densities),
            ("features", self.features),
        ):
            if value.shape[0] != n_points:
                raise ValueError(
                    f"{name} has {value.shape[0]} points, expected {n_points}."
                )
        if self.positions.shape[-1] != 3:
            raise ValueError(f"positions must have shape [N, 3], got {self.positions.shape}.")
        if self.rotations.shape[-1] != 4:
            raise ValueError(f"rotations must have shape [N, 4], got {self.rotations.shape}.")
        if self.scales.shape[-1] != 3:
            raise ValueError(f"scales must have shape [N, 3], got {self.scales.shape}.")
        if self.densities.shape[-1] != 1:
            raise ValueError(f"densities must have shape [N, 1], got {self.densities.shape}.")
        return self
