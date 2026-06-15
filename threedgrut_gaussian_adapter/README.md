# 3DGRUT Gaussian Adapter

This package exposes a portable rendering boundary for external 3D Gaussian systems. The caller owns dataset loading, checkpoint loading, deformation evaluation, and activation of Gaussian parameters. The adapter owns camera-ray batch creation and calls into the 3DGRT/3DGUT tracer tensor APIs.

## Static Render

```python
from threedgrut_gaussian_adapter import CameraState, GaussianState, GrutRenderer, RenderEffects

gaussians = GaussianState(
    positions=positions,
    rotations=rotations,
    scales=scales,
    densities=densities,
    features=features,
    active_sh_degree=active_sh_degree,
)
camera = CameraState(width=w, height=h, fovx=fovx, fovy=fovy, c2w=c2w)
renderer = GrutRenderer(mode="3dgrt", config=tracer_config)
outputs = renderer.render(gaussians, camera, effects=RenderEffects())
```

Returned outputs include `render`, `depth`, `opacity`, and `means3Dfinal`.

## Temporal Render

For rolling shutter or other temporal effects, provide a `state_at_time(t)` callback that returns an activated `GaussianState`. The adapter never evaluates external deformation networks itself.
