from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import OpacityResetDensificationTrainer
# from gaussian_splatting.trainer import BaseOpacityResetDensificationTrainer as OpacityResetDensificationTrainer
from gaussian_splatting.trainer import OpacityResetter, CameraTrainerWrapper
from .shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from .shculling import SHCullingTrainer
# from .shculling import BaseSHCullingTrainer as SHCullingTrainer
from .pruning import PruningTrainer, PrunerInDensifyTrainer
# from .pruning import BasePruningTrainer as PruningTrainer, BasePrunerInDensifyTrainer as PrunerInDensifyTrainer


def OpacityResetPruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        # these params are better than default ones
        mercy_type='redundancy_opacity_opacity',  # replace the default 'redundancy_opacity'. therefore work with opacity reset (prune low-opacity points)
        opacity_reset_interval=500,  # replace the default 100. since prune by opacity, donot prune too fast (train enough after opacity reset)
        prune_interval=500,  # replace the default 100. since prune by opacity, donot prune too fast (train enough after opacity reset)
        *args, **kwargs):
    return OpacityResetter(
        PruningTrainer(
            model, scene_extent, dataset,
            *args,
            prune_interval=prune_interval,
            mercy_type=mercy_type,
            **kwargs
        ),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )


def OpacityResetPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        # these params are better than default ones
        mercy_type='redundancy_opacity_opacity',  # replace the default 'redundancy_opacity'. therefore work with opacity reset (prune low-opacity points)
        opacity_reset_interval=500,  # replace the default 100. since prune by opacity, donot prune too fast (train enough after opacity reset)
        prune_interval=500,  # replace the default 100. since prune by opacity, donot prune too fast (train enough after opacity reset)
        *args, **kwargs):
    return OpacityResetter(
        PrunerInDensifyTrainer(
            model, scene_extent, dataset,
            *args,
            prune_interval=prune_interval,
            mercy_type=mercy_type,
            **kwargs
        ),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )


def SHCullingDensifyTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingPruneTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        OpacityResetPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingPruningDensifyTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        OpacityResetPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


class CameraTrainableVariableSHGaussianModel(VariableSHGaussianModel):
    def forward(self, camera: Camera):
        return CameraTrainableGaussianModel.forward(self, camera)


def CameraSHCullingTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        OpacityResetPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraOpacityResetPrunerInDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        OpacityResetPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingPruneTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingPruneTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingPruningDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingPruningDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
