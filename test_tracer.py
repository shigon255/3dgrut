"""
Minimal test to verify threedgrt_tracer compiles, runs, and supports backprop + optimization.
Tests:
  1. slangc compilation + JIT C++/CUDA build
  2. BVH construction
  3. Forward render pass
  4. Backward pass (gradient check)
  5. Optimization loop (loss should decrease)

NOTE: torch.cuda.synchronize() between build_acc_custom and render_custom is required because
the scene AABB is copied GPU→CPU asynchronously in buildBVH. Without sync, trace() may read
a zero/stale AABB and every ray misses.
"""
import sys
import math
import torch
from omegaconf import OmegaConf

render_conf = OmegaConf.create({
    "render": {
        "pipeline_type": "reference",
        "backward_pipeline_type": "referenceBwd",
        "primitive_type": "instances",
        "particle_kernel_degree": 4,
        "particle_kernel_min_response": 0.0113,
        "particle_kernel_min_alpha": 1.0 / 255.0,
        "particle_kernel_max_alpha": 0.99,
        "particle_kernel_density_clamping": False,
        "particle_radiance_sph_degree": 3,
        "min_transmittance": 0.001,
        "max_consecutive_bvh_update": 15,
        "enable_normals": False,
        "enable_hitcounts": True,
        "enable_kernel_timings": False,
    }
})

device = "cuda"
N = 1000
H, W = 64, 64
n_features = (render_conf.render.particle_radiance_sph_degree + 1) ** 2 * 3  # 48

from threedgrut.datasets.protocols import Batch

def make_batch():
    """Pinhole camera at origin looking along +z (right/down/front convention)."""
    fov_rad = math.radians(60)
    f = H / (2 * math.tan(fov_rad / 2))
    i = torch.arange(H, dtype=torch.float32, device=device) - H / 2 + 0.5
    j = torch.arange(W, dtype=torch.float32, device=device) - W / 2 + 0.5
    jj, ii = torch.meshgrid(j, i, indexing="xy")
    dirs = torch.stack([jj / f, ii / f, torch.ones_like(jj)], dim=-1).unsqueeze(0)  # [1,H,W,3] unnormalized
    rays_ori = torch.zeros(1, H, W, 3, device=device)
    T_to_world = torch.eye(4, device=device).unsqueeze(0)
    return Batch(rays_ori=rays_ori, rays_dir=dirs, T_to_world=T_to_world)

def make_gaussians(requires_grad=False):
    """1000 Gaussians clustered in front of the camera (z ≈ 3)."""
    pos = torch.randn(N, 3, device=device) * 0.5
    pos[:, 2] += 3.0
    rot = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=1)
    scl = torch.rand(N, 3, device=device) * 0.2 + 0.1
    dns = torch.full((N, 1), 3.0, device=device)
    feat = torch.zeros(N, n_features, device=device)
    if requires_grad:
        return (pos.requires_grad_(True), rot.detach().requires_grad_(True),
                scl.requires_grad_(True), dns.requires_grad_(True),
                feat.requires_grad_(True))
    return pos, rot, scl, dns, feat

def build_and_sync(tracer, pos, rot, scl, dns, rebuild=True):
    tracer.build_acc_custom(pos, rot, scl, dns, rebuild=rebuild)
    torch.cuda.synchronize()   # wait for async GPU→CPU gasAABB copy

# --------------------------------------------------------------------------
print("[1/5] Compiling threedgrt_tracer (slangc + JIT)...")
try:
    from threedgrt_tracer import Tracer
    tracer = Tracer(render_conf)
    print("      OK")
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
print("[2/5] Building BVH with synthetic Gaussians...")
try:
    positions, rotations, scales, densities, features = make_gaussians()
    build_and_sync(tracer, positions, rotations, scales, densities)
    print("      OK")
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
print("[3/5] Forward render pass...")
try:
    out = tracer.render_custom(
        positions=positions, rotations=rotations, scales=scales,
        densities=densities, features=features,
        n_active_features=n_features, gpu_batch=make_batch(),
    )
    mean_opacity = out["pred_opacity"].mean().item()
    hits = out["hits_count"].sum().item()
    print(f"      OK  —  pred_rgb: {out['pred_rgb'].shape}, "
          f"mean_opacity: {mean_opacity:.4f}, total_hits: {hits:.0f}")
    if mean_opacity < 1e-3:
        print("      WARNING: very low opacity — ray/Gaussian intersection may be failing")
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
print("[4/5] Backward pass (gradient check)...")
try:
    pos, rot, scl, dns, feat = make_gaussians(requires_grad=True)
    build_and_sync(tracer, pos.detach(), rot.detach(), scl.detach(), dns.detach())
    out = tracer.render_custom(
        positions=pos, rotations=rot, scales=scl,
        densities=dns, features=feat,
        n_active_features=n_features, gpu_batch=make_batch(),
    )
    loss = out["pred_rgb"].mean()
    loss.backward()

    grads = {"positions": pos.grad, "rotations": rot.grad, "scales": scl.grad,
             "densities": dns.grad, "features": feat.grad}
    missing = [k for k, g in grads.items() if g is None]
    zero    = [k for k, g in grads.items() if g is not None and g.abs().max() == 0]

    if missing:
        print(f"      FAILED: no gradient for: {missing}")
        sys.exit(1)
    if zero:
        print(f"      WARNING: zero gradient for: {zero}")
    else:
        print("      OK  —  all parameters received non-zero gradients")
    for k, g in grads.items():
        if g is not None:
            print(f"             {k:12s}  grad max={g.abs().max():.4e}")
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------
print("[5/5] Optimization loop (10 steps, loss should decrease)...")
try:
    target = torch.ones(1, H, W, 3, device=device)

    pos, rot, scl, dns, feat = make_gaussians(requires_grad=True)
    optimizer = torch.optim.Adam([pos, rot, scl, dns, feat], lr=1e-2)

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        with torch.no_grad():
            rot.data = torch.nn.functional.normalize(rot.data, dim=1)
        build_and_sync(tracer, pos.detach(), rot.detach(), scl.detach(), dns.detach())
        out = tracer.render_custom(
            positions=pos, rotations=rot, scales=scl,
            densities=dns, features=feat,
            n_active_features=n_features, gpu_batch=make_batch(),
        )
        loss = torch.nn.functional.mse_loss(out["pred_rgb"], target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"             step {step+1:2d}  loss={loss.item():.6f}  "
              f"opacity={out['pred_opacity'].mean().item():.4f}")

    if losses[-1] < losses[0]:
        print(f"      OK  —  loss decreased {losses[0]:.6f} → {losses[-1]:.6f}")
    else:
        print(f"      FAILED: loss did not decrease ({losses[0]:.6f} → {losses[-1]:.6f})")
        sys.exit(1)
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

print("\nAll tests passed.")
