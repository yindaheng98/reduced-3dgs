import math
from typing import Callable, List
import torch

from gaussian_splatting import Camera, GaussianModel
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


def prune_gaussians(model: GaussianModel, dataset: CameraDataset):
    gaussian_count = torch.zeros(model.get_xyz.shape[0], device=model.get_xyz.device, dtype=torch.int)
    opacity_important_score = torch.zeros(model.get_xyz.shape[0], device=model.get_xyz.device, dtype=torch.float)
    T_alpha_important_score = torch.zeros(model.get_xyz.shape[0], device=model.get_xyz.device, dtype=torch.float)
    for camera in dataset:
        out = count_render(model, camera)
        gaussian_count += out["gaussians_count"]
        opacity_important_score += out["opacity_important_score"]
        T_alpha_important_score += out["T_alpha_important_score"]
    return None


class ImportancePruner(DensifierWrapper):
    def __init__(
            self, base_densifier: AbstractDensifier,
            dataset: CameraDataset,
            importance_prune_from_iter=15000,
            importance_prune_until_iter=20000,
            importance_prune_interval: int = 1000,
    ):
        super().__init__(base_densifier)
        self.dataset = dataset
        self.importance_prune_from_iter = importance_prune_from_iter
        self.importance_prune_until_iter = importance_prune_until_iter
        self.importance_prune_interval = importance_prune_interval

    def densify_and_prune(self, loss, out, camera, step: int):
        ret = super().densify_and_prune(loss, out, camera, step)
        if self.importance_prune_from_iter <= step <= self.importance_prune_until_iter and step % self.importance_prune_interval == 0:
            remove_mask = prune_gaussians(self.model, self.dataset)
            ret = ret._replace(remove_mask=remove_mask if ret.remove_mask is None else torch.logical_or(ret.remove_mask, remove_mask))
        return ret


def BaseImportancePruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=1000,
        importance_prune_until_iter=15000,
        importance_prune_interval: int = 100,
        **kwargs):
    return DensificationTrainer(
        model, scene_extent,
        ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
        ), *args, **kwargs
    )
