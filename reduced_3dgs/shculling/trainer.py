from typing import List
import torch

from gaussian_splatting import Camera
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, BaseTrainer
from gaussian_splatting.dataset import CameraDataset
from reduced_3dgs.diff_gaussian_rasterization._C import calculate_colours_variance

from .gaussian_model import VariableSHGaussianModel


def _low_variance_colour_culling(self: VariableSHGaussianModel, threshold, weighted_variance: torch.Tensor, weighted_mean: torch.Tensor):
    original_degrees = torch.zeros_like(self._degrees)
    original_degrees.copy_(self._degrees)

    # Uniform colour culling
    weighted_colour_std = weighted_variance.sqrt()
    weighted_colour_std[weighted_colour_std.isnan()] = 0
    weighted_colour_std = weighted_colour_std.mean(dim=2).squeeze()

    std_mask = weighted_colour_std < threshold
    self._features_dc[std_mask] = (weighted_mean[std_mask] - 0.5) / 0.28209479177387814
    self._degrees[std_mask] = 0
    self._features_rest[std_mask] = 0


def _low_distance_colour_culling(self: VariableSHGaussianModel, threshold, colour_distances: torch.Tensor):
    colour_distances[colour_distances.isnan()] = 0

    # Loop from active_sh_degree - 1 to 0, since the comparisons
    # are always done based on the max band that corresponds to active_sh_degree
    for sh_degree in range(self.active_sh_degree - 1, 0, -1):
        coeffs_num = (sh_degree+1)**2 - 1
        mask = colour_distances[:, sh_degree] < threshold
        self._degrees[mask] = torch.min(
            torch.tensor([sh_degree], device="cuda", dtype=int),
            self._degrees[mask]
        ).int()

        # Zero-out the associated SH coefficients for clarity,
        # as they won't be used in rasterisation due to the degrees field
        self._features_rest[mask, coeffs_num:] = 0


def _cull_sh_bands(self: VariableSHGaussianModel, cameras: List[Camera], threshold=0, std_threshold=0.):
    camera_positions = torch.stack([cam.camera_center for cam in cameras], dim=0)
    camera_viewmatrices = torch.stack([cam.world_view_transform for cam in cameras], dim=0)
    camera_projmatrices = torch.stack([cam.full_proj_transform for cam in cameras], dim=0)
    camera_fovx = torch.tensor([camera.FoVx for camera in cameras], device="cuda", dtype=torch.float32)
    camera_fovy = torch.tensor([camera.FoVy for camera in cameras], device="cuda", dtype=torch.float32)
    image_height = torch.tensor([camera.image_height for camera in cameras], device="cuda", dtype=torch.int32)
    image_width = torch.tensor([camera.image_width for camera in cameras], device="cuda", dtype=torch.int32)

    # Wrapping in a function since it's called with the same parameters twice
    def run_calculate_colours_variance():
        return calculate_colours_variance(
            camera_positions,
            self.get_xyz,
            self._opacity,
            self.get_scaling,
            self.get_rotation,
            camera_viewmatrices,
            camera_projmatrices,
            torch.tan(camera_fovx*0.5),
            torch.tan(camera_fovy*0.5),
            image_height,
            image_width,
            self.get_features,
            self._degrees,
            self.active_sh_degree)

    _, weighted_variance, weighted_mean = run_calculate_colours_variance()
    _low_variance_colour_culling(self, std_threshold, weighted_variance, weighted_mean)

    # Recalculate to account for the changed values
    colour_distances, _, _ = run_calculate_colours_variance()
    _low_distance_colour_culling(self, threshold, colour_distances)


def cull_sh_bands(self: VariableSHGaussianModel, cameras: List[Camera], threshold=0, std_threshold=0.):
    with torch.no_grad():
        _cull_sh_bands(self, cameras, threshold, std_threshold)


class SHCuller(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            dataset: CameraDataset,
            cdist_threshold: float = 6,
            std_threshold: float = 0.04,
            cull_at_steps=[15000],
    ):
        super().__init__(base_trainer)
        assert isinstance(self.model, VariableSHGaussianModel)
        self.dataset = dataset
        self.cdist_threshold = cdist_threshold
        self.std_threshold = std_threshold
        self.cull_at_steps = cull_at_steps

    def optim_step(self):
        ret = super().optim_step()
        if self.curr_step in self.cull_at_steps:
            cull_sh_bands(self.model, self.dataset, self.cdist_threshold, self.std_threshold)
        return ret


def BaseSHCullingTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        BaseTrainer(model, scene_extent, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )
