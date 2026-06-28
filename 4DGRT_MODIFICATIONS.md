# 4DGRT-Specific Modifications

This file documents changes made to this fork for the 4DGRT project on top of the
upstream `nv-tlabs/3dgrut` release.

## Blender Fisheye Camera Model for 3DGUT

**Commits**: `2cc94cd`, `8e6c768`

Added `BlenderFisheyeProjectionParameters` to the 3DGUT tiling kernel so the Blender
lens-polynomial fisheye model (`θ = k0 + k1·r_mm + … + k4·r_mm⁴`, r in mm, θ in radians)
can be used end-to-end for both per-pixel ray generation and UT Gaussian tiling/projection.

The upstream 3DGUT only supported the OpenCV fisheye model
(`θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)`). Using it with Blender-polynomial rays caused a
model mismatch: rays and tiling projected Gaussians to different screen positions, producing
whole-image tile-aligned block artifacts in `--effect fisheye` renders.

**Files changed**:
- `threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh`
  - `projectPoint(BlenderFisheyeProjectionParameters)`: Newton-iteration inversion of
    `θ = poly(r_mm)` to get r_mm, then `pixel = pp + pixelsPerMm * r_mm * dir`
  - Returns `valid = (thetaFull < maxAngle)` — see UT fix below for why withinResolution
    is excluded
- `threedgut_tracer/include/3dgut/sensors/cameraModels.h`
  - `BlenderFisheyeProjectionParameters` struct (`principalPoint`, `pixelsPerMm`,
    `radialCoeffs[5]`, `maxAngle`)
  - `BlenderFisheyeModel` enum value in `TSensorModel`
- `threedgut_tracer/bindings.cpp`
  - `fromBlenderFisheyeCameraModelParameters(resolution, shutter_type, principal_point,
    radial_coeffs, sensor_width_mm, sensor_height_mm, fisheye_fov_deg)` Python binding
  - Derives `pixelsPerMm = W / adj_sw` with aspect-ratio normalization
  - Derives `maxAngle = fisheye_fov_deg * π / 180 / 2`

---

## 3DGUT UT Sigma-Point Fix for Fisheye

**Commit**: `a8ed74e`

Fixed 16×16 tile-block artifacts spread across the **entire** fisheye image (not just
periphery).

### Root Cause

3D Gaussians with large depth extent (`scale_z`) have UT sigma points offset along the
depth axis. The `-z` sigma point can land **behind the camera** (`z < 0`,
`theta >= maxAngle = 90°`). The kernel clamped these to the FOV boundary circle
(`r_mm` at `maxAngle`) and included them in the UT covariance estimate (include-all mode).
All behind-camera sigma points collapse to the same boundary ring, distorting the projected
ellipse and causing wrong tile assignments and K-buffer overflow throughout the image.

### Fix

Two coordinated changes:

**1. `cameraProjections.cuh` — Blender fisheye `projectPoint` validity:**

```cpp
// Before:
return (thetaFull < sensorParams.maxAngle)
       && withinResolution({(float)resolution.x, (float)resolution.y}, tolerance, projected);

// After:
return (thetaFull < sensorParams.maxAngle);
```

`valid = true` now means "in front of camera" only. Sigma points with `theta < maxAngle`
that fall outside the image boundary already have **correct extrapolated positions** (no
theta clamping is applied when `thetaFull < maxAngle`), so they must be included in the UT
covariance estimate — not excluded via `withinResolution`.

**2. `gutProjector.cuh` — UT center and covariance use valid sigma points only:**

```
particleProjCenter    = weighted mean of validSigmaPoints only (renormalize weights)
particleProjCovariance = weighted sum of outer products of validSigmaPoints only
```

Invalid = behind-camera (`theta >= maxAngle`). These are excluded because their
boundary-circle positions bias the covariance estimate.

### Why the Earlier Exclusion Attempt (commit `72a95fb`) Hurt Quality

Commit `72a95fb` applied sigma-point exclusion but kept `valid = theta < maxAngle AND
withinResolution`. For the Blender fisheye model with `maxAngle = 90°`, the image only
extends to ~45° at the corners, so many in-front-of-camera sigma points fall outside
`withinResolution` at moderate field angles. Excluding those underestimated the projected
Gaussian footprint → missing tile coverage → −1.5 dB PSNR.

The corrected fix separates the two conditions:
- `withinResolution = false` but `theta < maxAngle`: **include** (correct extrapolated pos)
- `theta >= maxAngle` (behind camera): **exclude** (clamped boundary-circle pos is wrong)

---

## Other Fixes

| Commit | Description |
|--------|-------------|
| `2d2deaa` | Fix OpenCV fisheye `radial_coeffs` truncated to 4 coeffs for 3DGUT |
| `79d0331` | Fix Slang pointer syntax for slangc 2024.17 / 2025.x compatibility |
| `32502df` | Fix Slang `Access.Read` undefined identifier with slangc 2024.17 |
| `cdf6cf7` | Integrate 4DGRT rendering adapter updates |
| `d24faf7` | Add portable Gaussian adapter |
