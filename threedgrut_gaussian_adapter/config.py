from argparse import ArgumentParser, ArgumentTypeError, Namespace

from omegaconf import OmegaConf


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ArgumentTypeError("Boolean value expected.")


def add_3dgrt_params(parser: ArgumentParser):
    group = parser.add_argument_group("3DGRT Parameters")
    group.add_argument("--render.method", type=str, default="3dgrt", help="3dgrt or 3dgut")
    group.add_argument("--render.pipeline_type", type=str, default="reference")
    group.add_argument("--render.backward_pipeline_type", type=str, default="referenceBwd")
    group.add_argument("--render.particle_kernel_degree", type=int, default=4)
    group.add_argument("--render.particle_kernel_density_clamping", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.particle_kernel_min_response", type=float, default=0.0113)
    group.add_argument("--render.particle_kernel_min_alpha", type=float, default=1.0 / 255.0)
    group.add_argument("--render.particle_kernel_max_alpha", type=float, default=0.99)
    group.add_argument("--render.particle_radiance_sph_degree", type=int, default=3)
    group.add_argument("--render.primitive_type", type=str, default="instances")
    group.add_argument("--render.min_transmittance", type=float, default=0.001)
    group.add_argument("--render.max_consecutive_bvh_update", type=int, default=15)
    group.add_argument("--render.enable_normals", type=str2bool, nargs="?", const=False, default=False)
    group.add_argument("--render.enable_hitcounts", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.enable_kernel_timings", type=str2bool, nargs="?", const=False, default=False)


def add_3dgut_params(parser: ArgumentParser):
    group = parser.add_argument_group("3DGUT Parameters")
    group.add_argument("--render.method", type=str, default="3dgut", help="3dgut")
    group.add_argument("--render.pipeline_type", type=str, default="reference")
    group.add_argument("--render.backward_pipeline_type", type=str, default="referenceBwd")
    group.add_argument("--render.particle_kernel_density_clamping", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.particle_kernel_min_alpha", type=float, default=1.0 / 255.0)
    group.add_argument("--render.particle_kernel_max_alpha", type=float, default=0.99)
    group.add_argument("--render.particle_radiance_sph_degree", type=int, default=3)
    group.add_argument("--render.primitive_type", type=str, default="instances")
    group.add_argument("--render.max_consecutive_bvh_update", type=int, default=15)
    group.add_argument("--render.enable_normals", type=str2bool, nargs="?", const=False, default=False)
    group.add_argument("--render.enable_hitcounts", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.enable_kernel_timings", type=str2bool, nargs="?", const=False, default=False)
    group.add_argument("--render.particle_kernel_degree", type=int, default=2, help="Degree of the particle kernel")
    group.add_argument("--render.particle_kernel_min_response", type=float, default=0.0)
    group.add_argument("--render.min_transmittance", type=float, default=0.0001)
    group.add_argument("--render.splat.rect_bounding", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.splat.tight_opacity_bounding", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.splat.tile_based_culling", type=str2bool, nargs="?", const=True, default=True)
    group.add_argument("--render.splat.extent_factor_cap", type=float, default=3.33, help="Cap on the projected-extent safety-margin multiplier. Raise for strongly nonlinear (e.g. fisheye) projections where thin/anisotropic Gaussians' true curved footprint can exceed the default cap.")
    group.add_argument("--render.splat.n_rolling_shutter_iterations", type=int, default=5)
    group.add_argument("--render.splat.ut_alpha", type=float, default=1.0)
    group.add_argument("--render.splat.ut_beta", type=float, default=2.0)
    group.add_argument("--render.splat.ut_kappa", type=float, default=0.0)
    group.add_argument("--render.splat.ut_in_image_margin_factor", type=float, default=0.1)
    group.add_argument("--render.splat.ut_require_all_sigma_points_valid", type=str2bool, nargs="?", const=False, default=False)
    group.add_argument("--render.splat.k_buffer_size", type=int, default=0)
    group.add_argument("--render.splat.global_z_order", type=str2bool, nargs="?", const=True, default=True)


def namespace_to_dict(ns: Namespace):
    result = {}
    for key in dir(ns):
        if key.startswith("__"):
            continue
        value = getattr(ns, key)
        if isinstance(value, Namespace):
            result[key] = namespace_to_dict(value)
        elif not callable(value):
            result[key] = value
    return result


def build_tracer_config(args):
    if getattr(args, "mode", None) not in ("3dgrt", "3dgut"):
        return None
    render_args = {
        key.replace("render.", ""): value
        for key, value in vars(args).items()
        if key.startswith("render.")
    }
    splat_args = None
    if args.mode == "3dgut":
        splat_args = {
            key.replace("splat.", ""): value
            for key, value in render_args.items()
            if key.startswith("splat.")
        }
    tracer_cfg = Namespace()
    tracer_cfg.render = Namespace(**render_args)
    tracer_cfg.render.splat = Namespace(**splat_args) if splat_args is not None else None
    return OmegaConf.create(namespace_to_dict(tracer_cfg))
