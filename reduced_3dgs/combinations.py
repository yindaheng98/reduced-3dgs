from typing import List
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import OpacityResetDensificationTrainer
# from gaussian_splatting.trainer import BaseOpacityResetDensificationTrainer as OpacityResetDensificationTrainer
from gaussian_splatting.trainer import OpacityResetTrainerWrapper, CameraTrainerWrapper, NoopDensifier, DepthTrainerWrapper
from .shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from .shculling import SHCullingTrainer
# from .shculling import BaseSHCullingTrainer as SHCullingTrainer
from .pruning import PruningTrainerWrapper, PrunerInDensifyTrainerWrapper
# from .pruning import BasePruningTrainer as PruningTrainer, BasePrunerInDensifyTrainer as PrunerInDensifyTrainer
from .importance import ImportancePruner


def BaseFullPruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=1000,
        importance_prune_until_iter=15000,
        importance_prune_interval: int = 100,
        **kwargs):
    return PruningTrainerWrapper(
        lambda model, scene_extent, dataset: ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
        ),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseFullPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=1000,
        importance_prune_until_iter=15000,
        importance_prune_interval: int = 100,
        **kwargs):
    return PrunerInDensifyTrainerWrapper(
        lambda model, scene_extent, dataset: ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
        ),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthFullPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseFullPruningTrainer, model, scene_extent, *args, dataset=dataset, **kwargs)


def DepthFullPrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseFullPrunerInDensifyTrainer, model, scene_extent, *args, dataset=dataset, **kwargs)


def OpacityResetPruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return OpacityResetTrainerWrapper(
        lambda model, scene_extent, *args, **kwargs: DepthFullPruningTrainer(model, scene_extent, dataset, *args, **kwargs),
        model, scene_extent,
        *args, **kwargs
    )


def OpacityResetPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return OpacityResetTrainerWrapper(
        lambda model, scene_extent, *args, **kwargs: DepthFullPrunerInDensifyTrainer(model, scene_extent, dataset, *args, **kwargs),
        model, scene_extent,
        *args, **kwargs
    )


PruningTrainer = OpacityResetPruningTrainer
PrunerInDensifyTrainer = OpacityResetPrunerInDensifyTrainer


def SHCullingDensificationTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingPruningTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        OpacityResetPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingPrunerInDensifyTrainer(
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


def CameraPrunerInDensifyTrainer(
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
        SHCullingDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingPruningDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
