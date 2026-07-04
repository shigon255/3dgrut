# 4DGRT-Specific Modifications

This file documents changes made to this fork for the 4DGRT project on top of the
upstream `nv-tlabs/3dgrut` release.

> **Note on the 3DGUT fisheye tile-block artifact**: several fixes below (the UT sigma-point
> fix, the maxAngle fix) were investigated as causes of the artifact but turned out to be
> no-ops or minor contributors at best. Jump to
> [**"3DGUT Fisheye Tile-Boundary Artifact: Actual Root Cause and Fix"**](#3dgut-fisheye-tile-boundary-artifact-actual-root-cause-and-fix)
> below for the fix that actually resolves it.

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

## 3DGUT Fisheye Tile-Boundary Artifact: Actual Root Cause and Fix

**Commits**: `5a85036` (superseded/no-op, see below), `e273fe9`, `74913e4`

The UT sigma-point fix above (`a8ed74e`) and the maxAngle fix (`5a85036`) were both
investigated as fixes for 16×16 tile-block artifacts, but **neither was the real cause**.
Measured before/after pixel diffs showed `5a85036` changed less than 0.01% of pixels by
at most 1 gray level — a no-op in practice. The actual root cause and fix, found by
systematically testing each 3DGUT tiling/culling approximation in isolation and measuring
a 16px tile-boundary discontinuity ratio (mean gradient magnitude at tile-boundary rows/cols
vs. elsewhere; ~1.0 = artifact-free, matching GT and 3DGRT), is documented here.

### Root Cause

3DGUT's tiling step estimates each Gaussian's projected 2D covariance (screen-space ellipse)
using the Unscented Transform (UT) — 7 sigma points sampled near the Gaussian's mean. This is
a **local linearization**: it implicitly assumes the camera projection is close to affine in
the neighborhood of the Gaussian. For pinhole cameras this holds almost exactly. For fisheye
(strongly nonlinear, increasingly curved at wide field angles), the 7-point estimate
systematically **underestimates** the Gaussian's true (ray-traced) screen footprint, so some
tiles that should include the Gaussian never get it added to their candidate list —
independent of any subsequent per-tile culling.

Three optimizations in the tiling path — safe/nearly-free for pinhole cameras, where the UT
estimate is already accurate — compound this underestimation for fisheye:

- `render.splat.tile_based_culling` (`GAUSSIAN_TILE_BASED_CULLING`): per-tile pruning based on
  a minimum-power-response test over the (already too-small) bounding box.
- `render.splat.rect_bounding` (`GAUSSIAN_RECT_BOUNDING`): uses the smaller of an axis-aligned
  bound vs. the full isotropic radius.
- `render.splat.tight_opacity_bounding` (`GAUSSIAN_TIGHT_OPACITY_BOUNDING`): shrinks the extent
  safety-margin multiplier (`extentFactor`) adaptively based on opacity instead of using the
  fixed conservative cap.

The severity scales with the **physical size of the Gaussian**: a large, isotropic Gaussian
(e.g. fit to a round object) spans much more projection curvature across its own extent than a
thin, flat-surface Gaussian, so its UT covariance estimate is proportionally worse. This is
also much worse when Gaussians are optimized on **pinhole** training data and only rendered
through the fisheye model afterward: under pinhole, the UT estimate is accurate regardless of
Gaussian size/shape, so training gives the optimizer no incentive to avoid large, round
Gaussians — the failure mode is only exposed at fisheye render time. (Official 3DGUT usage,
e.g. ScanNet++, trains and tests on the same fisheye/wide-FOV camera model, so the optimizer
adapts Gaussian shapes to keep the UT approximation accurate throughout training.)

### Fix

Four `render.splat.*` settings, tested individually via the same tile-boundary ratio metric
on real (pinhole-trained, fisheye-rendered) scenes:

| Setting | Default | Fisheye value | Measured effect |
|---|---|---|---|
| `tile_based_culling` | `true` | `false` | Largest single fix: ratio 2.82→~2.0 |
| `rect_bounding` | `true` | `false` | Second fix (combined w/ tight_opacity_bounding): 2.0→1.22 |
| `tight_opacity_bounding` | `true` | `false` | (see above, tested together) |
| `extent_factor_cap` (**new**, commit `74913e4`) | `3.33` | `6.0` | Closes remaining gap: 1.22→1.07 (GT baseline ~0.95) |

`extent_factor_cap` is a new config knob (`threedgrut_gaussian_adapter/config.py`,
`threedgut_tracer/setup_3dgut.py` → `-DGAUSSIAN_EXTENT_FACTOR_CAP`, `threedgut.cuh` →
`TGUTProjectorParams::ExtentFactorCap`, used in `gutProjector.cuh`'s
`computeProjectedExtentConicOpacity`) replacing the previously-hardcoded `3.33f` safety-margin
cap. Default preserves original behavior for all other camera models/scenes; only the fisheye
render path (wired in 4DGRT's `4DGaussians/render_scene.sh`) sets it to `6.0`.

Also tested and found **not** to help (kept at defaults): `global_z_order=false`, `k_buffer_size
=32` (both negligible effect — ruled out per-ray blend-order as a contributor), `ut_alpha=2.0`
(wider UT sigma-point spread — made the ratio *worse*, 1.22→1.26).

### Residual

Even with all four settings applied, a small residual remains (ratio ~1.07 vs. GT's ~0.95),
most visible on large/round objects (e.g. a ball Gaussian cluster showed the sharpest residual
artifact — a smeared, tile-aligned distortion — while flat wall surfaces are visually
indistinguishable from GT). This is consistent with the root cause: raising `extent_factor_cap`
widens the safety margin around an already-approximate covariance estimate, but doesn't fix the
estimate's shape itself. Training directly on fisheye data (rather than pinhole) is expected to
reduce this further by letting the optimizer avoid Gaussian shapes that are pathological under
the UT/fisheye approximation — not yet verified in this codebase.

### Unrelated Build Bug Fixed Along the Way (commit `e273fe9`)

While testing `k_buffer_size > 0` (per-ray depth-buffered blending, as a candidate fix), builds
failed with `static_assert failed with "evalForwardNoKBufferBalanced only supports K=0"` even
though `fine_grained_load_balancing` was `false`. Cause: `gutRenderer.cu`'s launch call for the
`renderBalanced` kernel was correctly guarded with `#if FINE_GRAINED_LOAD_BALANCING`, but the
kernel **definition** in `gutRenderer.cuh` was not — so it was always compiled (and its internal
`static_assert(KHitBufferSize==0)` always evaluated) regardless of whether it was ever launched.
Fixed by wrapping the definition in the same `#if` guard used at the call site. Unrelated to
fisheye; a latent bug in the pre-existing fine-grained load-balancing kernel.

---

## Other Fixes

| Commit | Description |
|--------|-------------|
| `2d2deaa` | Fix OpenCV fisheye `radial_coeffs` truncated to 4 coeffs for 3DGUT |
| `79d0331` | Fix Slang pointer syntax for slangc 2024.17 / 2025.x compatibility |
| `32502df` | Fix Slang `Access.Read` undefined identifier with slangc 2024.17 |
| `cdf6cf7` | Integrate 4DGRT rendering adapter updates |
| `d24faf7` | Add portable Gaussian adapter |
