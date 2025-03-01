from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import OpacityResetDensificationTrainer, OpacityResetter
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCuller
from reduced_3dgs.pruning import BasePruningTrainer, BasePrunerInDensifyTrainer
from reduced_3dgs.shculling import VariableSHGaussianModel


def OpacityResetPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        opacity_reset_interval=3000,
        *args, **kwargs):
    return OpacityResetter(
        BasePrunerInDensifyTrainer(
            model, scene_extent, dataset,
            *args, **kwargs
        ),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )


def SHCullingDensifyTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )


def SHCullingPruneTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        BasePruningTrainer(model, scene_extent, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )


def SHCullingPruningDensifyTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        OpacityResetPrunerInDensifyTrainer(model, scene_extent, dataset, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )
