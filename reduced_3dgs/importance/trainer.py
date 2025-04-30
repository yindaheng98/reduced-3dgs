import math
from typing import List
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.camera import build_camera
from gaussian_splatting.trainer import AbstractDensifier, DensifierWrapper, DensificationTrainer, NoopDensifier
from gaussian_splatting.dataset import CameraDataset
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def count_render(self: GaussianModel, viewpoint_camera: Camera):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device=self._xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=viewpoint_camera.bg_color.to(self._xyz.device),
        scale_modifier=self.scale_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=self.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=self.debug,
        f_count=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = self.get_xyz
    means2D = screenspace_points
    opacity = self.get_opacity

    scales = self.get_scaling
    rotations = self.get_rotation

    shs = self.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    gaussians_count, opacity_important_score, T_alpha_important_score, rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "gaussians_count": gaussians_count,
        "opacity_important_score": opacity_important_score,
        "T_alpha_important_score": T_alpha_important_score
    }


def prune_list(model: GaussianModel, dataset: CameraDataset, resize=None):
    gaussian_count = torch.zeros(model.get_xyz.shape[0], device=model.get_xyz.device, dtype=torch.int)
    opacity_important_score = torch.zeros(model.get_xyz.shape[0], device=model.get_xyz.device, dtype=torch.float)
    T_alpha_important_score = torch.zeros(model.get_xyz.shape[0], device=model.get_xyz.device, dtype=torch.float)
    for camera in dataset:
        if resize is not None:
            height, width = camera.image_height, camera.image_width
            scale = resize / max(height, width)
            height, width = int(height * scale), int(width * scale)
            camera = build_camera(
                image_height=height, image_width=width,
                FoVx=camera.FoVx, FoVy=camera.FoVy,
                R=camera.R, T=camera.T,
                device=camera.R.device)
        out = count_render(model, camera)
        gaussian_count += out["gaussians_count"]
        opacity_important_score += out["opacity_important_score"]
        T_alpha_important_score += out["T_alpha_important_score"]
    return gaussian_count, opacity_important_score, T_alpha_important_score


# return importance score with adaptive volume measure described in paper
def calculate_v_imp_score(gaussians: GaussianModel, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list


def score2mask(percent, import_score: list, threshold=None):
    sorted_tensor, _ = torch.sort(import_score, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    thr = min(threshold, value_nth_percentile) if threshold is not None else value_nth_percentile
    prune_mask = (import_score <= thr)
    return prune_mask


def prune_gaussians(
        gaussians: GaussianModel, dataset: CameraDataset,
        resize=None,
        prune_type="comprehensive",
        prune_percent=0.1,
        prune_thr_important_score=None,
        prune_thr_v_important_score=None,
        prune_thr_max_v_important_score=None,
        prune_thr_count=None,
        prune_thr_T_alpha=None,
        prune_thr_T_alpha_avg=None,
        v_pow=0.1):
    gaussian_list, opacity_imp_list, T_alpha_imp_list = prune_list(gaussians, dataset, resize)
    match prune_type:
        case "important_score":
            mask = score2mask(prune_percent, opacity_imp_list, prune_thr_important_score)
        case "v_important_score":
            v_list = calculate_v_imp_score(gaussians, opacity_imp_list, v_pow)
            mask = score2mask(prune_percent, v_list, prune_thr_v_important_score)
        case "max_v_important_score":
            v_list = opacity_imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
            mask = score2mask(prune_percent, v_list, prune_thr_max_v_important_score)
        case "count":
            mask = score2mask(prune_percent, gaussian_list, prune_thr_count)
        case "T_alpha":
            # new importance score defined by doji
            mask = score2mask(prune_percent, T_alpha_imp_list, prune_thr_T_alpha)
        case "T_alpha_avg":
            v_list = T_alpha_imp_list / gaussian_list
            v_list[gaussian_list <= 0] = 0
            mask = score2mask(prune_percent, v_list, prune_thr_T_alpha_avg)
        case "comprehensive":
            mask = torch.zeros_like(gaussian_list, dtype=torch.bool)
            if prune_thr_important_score is not None:
                mask |= score2mask(prune_percent, opacity_imp_list, prune_thr_important_score)
            if prune_thr_v_important_score is not None:
                v_list = calculate_v_imp_score(gaussians, opacity_imp_list, v_pow)
                mask |= score2mask(prune_percent, v_list, prune_thr_v_important_score)
            if prune_thr_max_v_important_score is not None:
                v_list = opacity_imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
                mask |= score2mask(prune_percent, v_list, prune_thr_max_v_important_score)
            if prune_thr_count is not None:
                mask |= score2mask(prune_percent, gaussian_list, prune_thr_count)
            if prune_thr_T_alpha is not None:
                mask |= score2mask(prune_percent, T_alpha_imp_list, prune_thr_T_alpha)
            if prune_thr_T_alpha_avg is not None:
                v_list = T_alpha_imp_list / gaussian_list
                v_list[gaussian_list <= 0] = 0
                mask |= score2mask(prune_percent, v_list, prune_thr_T_alpha_avg)
        case _:
            raise Exception("Unsupportive prunning method")
    return mask


class ImportancePruner(DensifierWrapper):
    def __init__(
            self, base_densifier: AbstractDensifier,
            dataset: CameraDataset,
            importance_prune_from_iter=15000,
            importance_prune_until_iter=20000,
            importance_prune_interval: int = 1000,
            importance_score_resize=None,
            importance_prune_type="comprehensive",
            importance_prune_percent=0.1,
            importance_prune_thr_important_score=None,
            importance_prune_thr_v_important_score=3.0,
            importance_prune_thr_max_v_important_score=None,
            importance_prune_thr_count=1,
            importance_prune_thr_T_alpha=1,
            importance_prune_thr_T_alpha_avg=0.001,
            importance_v_pow=0.1):
        super().__init__(base_densifier)
        self.dataset = dataset
        self.importance_prune_from_iter = importance_prune_from_iter
        self.importance_prune_until_iter = importance_prune_until_iter
        self.importance_prune_interval = importance_prune_interval
        self.resize = importance_score_resize
        self.prune_percent = importance_prune_percent
        self.prune_thr_important_score = importance_prune_thr_important_score
        self.prune_thr_v_important_score = importance_prune_thr_v_important_score
        self.prune_thr_max_v_important_score = importance_prune_thr_max_v_important_score
        self.prune_thr_count = importance_prune_thr_count
        self.prune_thr_T_alpha = importance_prune_thr_T_alpha
        self.prune_thr_T_alpha_avg = importance_prune_thr_T_alpha_avg
        self.v_pow = importance_v_pow
        self.prune_type = importance_prune_type

    def densify_and_prune(self, loss, out, camera, step: int):
        ret = super().densify_and_prune(loss, out, camera, step)
        if self.importance_prune_from_iter <= step <= self.importance_prune_until_iter and step % self.importance_prune_interval == 0:
            remove_mask = prune_gaussians(
                self.model, self.dataset,
                self.resize,
                self.prune_type, self.prune_percent,
                self.prune_thr_important_score, self.prune_thr_v_important_score,
                self.prune_thr_max_v_important_score, self.prune_thr_count,
                self.prune_thr_T_alpha, self.prune_thr_T_alpha_avg, self.v_pow,
            )
            ret = ret._replace(remove_mask=remove_mask if ret.remove_mask is None else torch.logical_or(ret.remove_mask, remove_mask))
        return ret


def BaseImportancePruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=15000,
        importance_prune_until_iter=20000,
        importance_prune_interval: int = 1000,
        importance_score_resize=None,
        importance_prune_type="comprehensive",
        importance_prune_percent=0.1,
        importance_prune_thr_important_score=None,
        importance_prune_thr_v_important_score=3.0,
        importance_prune_thr_max_v_important_score=None,
        importance_prune_thr_count=1,
        importance_prune_thr_T_alpha=1.0,
        importance_prune_thr_T_alpha_avg=0.001,
        importance_v_pow=0.1,
        **kwargs):
    return DensificationTrainer(
        model, scene_extent,
        ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
            importance_score_resize=importance_score_resize,
            importance_prune_type=importance_prune_type,
            importance_prune_percent=importance_prune_percent,
            importance_prune_thr_important_score=importance_prune_thr_important_score,
            importance_prune_thr_v_important_score=importance_prune_thr_v_important_score,
            importance_prune_thr_max_v_important_score=importance_prune_thr_max_v_important_score,
            importance_prune_thr_count=importance_prune_thr_count,
            importance_prune_thr_T_alpha=importance_prune_thr_T_alpha,
            importance_prune_thr_T_alpha_avg=importance_prune_thr_T_alpha_avg,
            importance_v_pow=importance_v_pow,
        ), *args, **kwargs
    )
