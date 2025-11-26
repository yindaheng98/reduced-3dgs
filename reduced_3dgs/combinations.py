from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import OpacityResetDensificationTrainer, OpacityResetTrainerWrapper, CameraTrainerWrapper, DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier, DensificationTrainer
from .shculling import VariableSHGaussianModel, SHCullingTrainerWrapper, SHCullingTrainer
from .pruning import PruningDensifierWrapper, ReducedDensificationDensifierWrapper
from .importance import ImportancePruningDensifierWrapper


# Full Pruning Trainer

def FullPruningDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs) -> AbstractDensifier:
    return PruningDensifierWrapper(
        partial(ImportancePruningDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def FullPruningTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(FullPruningDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseFullPruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return FullPruningTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthFullPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseFullPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


FullPruningTrainer = DepthFullPruningTrainer


# Full Reduced Densification Trainer

def FullReducedDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs) -> AbstractDensifier:
    return ReducedDensificationDensifierWrapper(
        partial(ImportancePruningDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def FullReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(FullReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseFullReducedDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return FullReducedDensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthFullReducedDensificationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


FullReducedDensificationTrainer = DepthFullReducedDensificationTrainer


# Full Reduced Densification Trainer + Opacity Reset

def OpacityResetFullReducedDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return OpacityResetTrainerWrapper(
        FullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


# SH Culling Wrapped Trainer

def SHCullingOpacityResetDensificationTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingFullPruningTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        FullPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingFullReducedDensificationTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        FullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def SHCullingOpacityResetFullReducedDensificationTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        OpacityResetFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


# Camera Wrapped Trainer

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


def CameraFullPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        FullPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        FullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraOpacityResetFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        OpacityResetFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


# Camera and SH Culling Wrapped Trainer

def CameraSHCullingOpacityResetDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingOpacityResetDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingFullPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingFullPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingOpacityResetFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingOpacityResetFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
