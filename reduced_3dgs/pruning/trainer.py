from typing import Callable, List
import torch
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.trainer import AbstractDensifier, OpacityPruner, DensificationTrainer, NoopDensifier
from reduced_3dgs.diff_gaussian_rasterization._C import sphere_ellipsoid_intersection, allocate_minimum_redundancy_value, find_minimum_projected_pixel_size
from reduced_3dgs.simple_knn._C import distIndex2


def calculate_redundancy_metric(gaussians: GaussianModel, cameras: List[Camera], pixel_scale=1.0, num_neighbours=30):
    # Get minimum projected pixel size
    cube_size = find_minimum_projected_pixel_size(
        torch.stack([camera.full_proj_transform for camera in cameras], dim=0),
        torch.stack([camera.full_proj_transform.inverse() for camera in cameras], dim=0),
        gaussians._xyz,
        torch.tensor([camera.image_height for camera in cameras], device="cuda", dtype=torch.int32),
        torch.tensor([camera.image_width for camera in cameras], device="cuda", dtype=torch.int32)
    )

    scaled_pixel_size = cube_size * pixel_scale
    half_diagonal = scaled_pixel_size * torch.sqrt(torch.tensor([3], device="cuda")) / 2

    # Find neighbours as candidates for the intersection test
    _, indices = distIndex2(gaussians.get_xyz, num_neighbours)
    indices = indices.view(-1, num_neighbours)

    # Do the intersection check
    redundancy_metrics, intersection_mask = sphere_ellipsoid_intersection(gaussians._xyz,
                                                                          gaussians.get_scaling,
                                                                          gaussians.get_rotation,
                                                                          indices,
                                                                          half_diagonal,
                                                                          num_neighbours)
    # We haven't counted count for the primitive at the center of each sphere, so add 1 to everything
    redundancy_metrics += 1

    indices = torch.cat((torch.arange(gaussians.get_xyz.shape[0], device="cuda", dtype=torch.int).view(-1, 1), indices), dim=1)
    intersection_mask = torch.cat((torch.ones_like(gaussians._opacity, device="cuda", dtype=bool), intersection_mask), dim=1)

    min_redundancy_metrics = allocate_minimum_redundancy_value(redundancy_metrics, indices, intersection_mask, num_neighbours+1)[0]
    return min_redundancy_metrics, cube_size


def mercy_points(self: GaussianModel, _splatted_num_accum: torch.Tensor, lambda_mercy=2, mercy_minimum=2, mercy_type='redundancy_opacity'):
    mean = _splatted_num_accum.float().mean(dim=0, keepdim=True)
    std = _splatted_num_accum.float().var(dim=0, keepdim=True).sqrt()

    threshold = max((mean + lambda_mercy*std).item(), mercy_minimum)

    mask = (_splatted_num_accum > threshold)

    if mercy_type == 'redundancy_opacity':
        # Prune redundant points based on 50% lowest opacity
        mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
    elif mercy_type == 'redundancy_random':
        # Prune 50% redundant points at random
        mask[mask.clone()] = torch.rand(mask[mask].shape, device="cuda").squeeze() < 0.5
    elif mercy_type == 'opacity':
        # Prune based just on opacity
        threshold = self.get_opacity.quantile(0.045)
        mask = (self.get_opacity < threshold).squeeze()
    elif mercy_type == 'redundancy_opacity_opacity':
        # Prune based on opacity and on redundancy + opacity (options 1 and 3)
        mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
        threshold = torch.min(self.get_opacity.quantile(0.03), torch.tensor([0.05], device="cuda"))
        mask = torch.logical_or(mask, (self.get_opacity < threshold).squeeze())
    return mask


def mercy_gaussians(
    model: GaussianModel,
    dataset: List[Camera],
    box_size=1.,
    lambda_mercy=1.,
    mercy_minimum=3,
    mercy_type='redundancy_opacity'
):
    _splatted_num_accum, _ = calculate_redundancy_metric(model, dataset, pixel_scale=box_size)
    mask = mercy_points(model, _splatted_num_accum.squeeze(), lambda_mercy, mercy_minimum, mercy_type)
    return mask


class BasePruner(OpacityPruner):
    def __init__(
            self, base_densifier: AbstractDensifier,
            scene_extent,
            dataset: List[Camera],
            *args,
            box_size=1.,
            lambda_mercy=1.,
            mercy_minimum=3,
            mercy_type='redundancy_opacity',
            **kwargs):
        super().__init__(base_densifier, scene_extent, *args, **kwargs)
        self.dataset = dataset
        self.box_size = box_size
        self.lambda_mercy = lambda_mercy
        self.mercy_minimum = mercy_minimum
        self.mercy_type = mercy_type

    def prune(self) -> torch.Tensor:
        remove_mask = mercy_gaussians(self.model, self.dataset, self.box_size, self.lambda_mercy, self.mercy_minimum, self.mercy_type)
        prune_mask = torch.logical_or(super().prune(), remove_mask)
        return prune_mask


def PruningTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float, List[Camera]], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval: int = 100,
        box_size=1.,
        lambda_mercy=1.,
        mercy_minimum=3,
        mercy_type='redundancy_opacity',
        **kwargs):
    return DensificationTrainer(
        model, scene_extent,
        BasePruner(
            noargs_base_densifier_constructor(model, scene_extent, dataset),
            scene_extent, dataset,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            box_size=box_size,
            lambda_mercy=lambda_mercy,
            mercy_minimum=mercy_minimum,
            mercy_type=mercy_type,
        ), *args, **kwargs
    )


def BasePruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args, **kwargs):
    return DensificationTrainer(
        lambda model, scene_extent, dataset: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )
