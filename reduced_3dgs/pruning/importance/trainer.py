import math
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, BaseTrainer, Trainer
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


class ImportancePruner(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            dataset: CameraDataset,
            importance_prune_at_steps=[15000],
    ):
        super().__init__(base_trainer)
        self.dataset = dataset
        self.importance_prune_at_steps = importance_prune_at_steps

    def optim_step(self):
        ret = super().optim_step()
        if self.curr_step in self.importance_prune_at_steps:
            gaussian_count = torch.zeros(self.model.get_xyz.shape[0], device=self.model.get_xyz.device, dtype=torch.int)
            opacity_important_score = torch.zeros(self.model.get_xyz.shape[0], device=self.model.get_xyz.device, dtype=torch.float)
            T_alpha_important_score = torch.zeros(self.model.get_xyz.shape[0], device=self.model.get_xyz.device, dtype=torch.float)
            for camera in self.dataset:
                out = count_render(self.model, camera)
                gaussian_count += out["gaussians_count"]
                opacity_important_score += out["opacity_important_score"]
                T_alpha_important_score += out["T_alpha_important_score"]
            pass
        return ret


def ImportancePruningTrainerWrapper(
    base_trainer_constructor,
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        importance_prune_at_steps=[15000],
        *args, **kwargs):
    return ImportancePruner(
        base_trainer_constructor(model, scene_extent, dataset, *args, **kwargs),
        dataset,
        importance_prune_at_steps=importance_prune_at_steps,
    )


def BaseImportancePruningTrainer(
    model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        importance_prune_at_steps=[15000],
        *args, **kwargs):
    return ImportancePruningTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        importance_prune_at_steps=importance_prune_at_steps,
        *args, **kwargs,
    )


def ImportancePruningTrainer(
    model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        importance_prune_at_steps=[15000],
        *args, **kwargs):
    return ImportancePruningTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: Trainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        importance_prune_at_steps=importance_prune_at_steps,
        *args, **kwargs,
    )
