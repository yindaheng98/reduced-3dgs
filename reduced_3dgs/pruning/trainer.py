from typing import List
import torch
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.trainer import AbstractDensifier, Densifier, DensificationInstruct, DensificationTrainer
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


class BasePruner(AbstractDensifier):
    def __init__(
            self, model: GaussianModel, dataset: List[Camera],
            prune_from_iter=1000,
            prune_until_iter=15000,
            prune_interval: int = 100,
            box_size=1.,
            lambda_mercy=1.,
            mercy_minimum=3,
            mercy_type='redundancy_opacity'):
        self.model = model
        self.dataset = dataset
        self.prune_from_iter = prune_from_iter
        self.prune_until_iter = prune_until_iter
        self.prune_interval = prune_interval
        self.box_size = box_size
        self.lambda_mercy = lambda_mercy
        self.mercy_minimum = mercy_minimum
        self.mercy_type = mercy_type

    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        if self.prune_from_iter < step < self.prune_until_iter and step % self.prune_interval == 0:
            return DensificationInstruct(mask=mercy_gaussians(self.model, self.dataset, self.box_size, self.lambda_mercy, self.mercy_minimum, self.mercy_type))
        return DensificationInstruct()


def BasePruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval: int = 100,
        box_size=1.,
        lambda_mercy=1.,
        mercy_minimum=3,
        mercy_type='redundancy_opacity',
        *args, **kwargs):
    return DensificationTrainer(
        model, scene_extent,
        BasePruner(
            model, dataset,
            prune_from_iter, prune_until_iter, prune_interval,
            box_size, lambda_mercy, mercy_minimum, mercy_type
        ), *args, **kwargs
    )


class PrunerInDensify(Densifier):
    def __init__(
            self, model: GaussianModel, scene_extent, dataset: List[Camera],
            mercy_from_iter=3000,
            mercy_until_iter=20000,
            mercy_interval: int = 100,
            box_size=1.,
            lambda_mercy=1.,
            mercy_minimum=3,
            mercy_type='redundancy_opacity',
            *args, **kwargs):
        super().__init__(model, scene_extent, *args, **kwargs)
        self.dataset = dataset
        self.mercy_from_iter = mercy_from_iter
        self.mercy_until_iter = mercy_until_iter
        self.mercy_interval = mercy_interval
        self.box_size = box_size
        self.lambda_mercy = lambda_mercy
        self.mercy_minimum = mercy_minimum
        self.mercy_type = mercy_type
        self.mercy_from_iter = mercy_from_iter

    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        instruct = super().densify_and_prune(loss, out, camera, step)
        if self.mercy_from_iter <= step < self.mercy_until_iter and step % self.mercy_interval == 0:
            _splatted_num_accum, _ = calculate_redundancy_metric(self.model, self.dataset, pixel_scale=self.box_size)
            mask = mercy_points(self.model, _splatted_num_accum.squeeze(), self.lambda_mercy, self.mercy_minimum, self.mercy_type)
            instruct = instruct._replace(remove_mask=mask if instruct.remove_mask is None else torch.logical_or(instruct.remove_mask, mask))
            self.xyz_gradient_accum = None
            self.denom = None
            self.max_radii2D = None
            torch.cuda.empty_cache()
        return instruct


def BasePrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,

        dataset: List[Camera],
        mercy_from_iter=3000,
        mercy_until_iter=20000,
        mercy_interval: int = 100,
        box_size=1.,
        lambda_mercy=1.,
        mercy_minimum=3,
        mercy_type='redundancy_opacity',

        percent_dense=0.01,

        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_opacity_threshold=0.005,

        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval=100,
        prune_screensize_threshold=20,

        *args, **kwargs):
    return DensificationTrainer(
        model, scene_extent,
        PrunerInDensify(
            model, scene_extent, dataset,
            mercy_from_iter, mercy_until_iter, mercy_interval,
            box_size, lambda_mercy, mercy_minimum, mercy_type,
            percent_dense,
            densify_from_iter, densify_until_iter, densify_interval, densify_grad_threshold, densify_opacity_threshold,
            prune_from_iter, prune_until_iter, prune_interval, prune_screensize_threshold
        ), *args, **kwargs
    )
