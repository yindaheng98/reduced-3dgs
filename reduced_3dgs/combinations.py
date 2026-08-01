from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import OpacityResetDensificationTrainer, OpacityResetTrainerWrapper, CameraTrainerWrapper, DepthTrainerWrapper, NormalTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier, DensificationTrainer
from .shculling import VariableSHGaussianModel, CameraTrainableVariableSHGaussianModel, SHCullingTrainerWrapper, SHCullingTrainer
from .pruning import PruningDensifierWrapper, ReducedDensificationDensifierWrapper
from .importance import ImportancePruningDensifierWrapper


# Full Pruning Trainer

def FullPruningDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs) -> AbstractDensifier:
    return PruningDensifierWrapper(
        partial(ImportancePruningDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def FullPruningTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(FullPruningDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def BaseFullPruningTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return FullPruningTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthFullPruningTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseFullPruningTrainer,
        model, dataset,
        **configs
    )


FullPruningTrainer = DepthFullPruningTrainer


# Full Reduced Densification Trainer

def FullReducedDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs) -> AbstractDensifier:
    return ReducedDensificationDensifierWrapper(
        partial(ImportancePruningDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def FullReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(FullReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def BaseFullReducedDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return FullReducedDensificationTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthFullReducedDensificationTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


FullReducedDensificationTrainer = DepthFullReducedDensificationTrainer


# Full Reduced Densification Trainer + Opacity Reset

def OpacityResetFullReducedDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return OpacityResetTrainerWrapper(
        FullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


# SH Culling Wrapped Trainer

def SHCullingOpacityResetDensificationTrainer(
    model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        OpacityResetDensificationTrainer,
        model, dataset,
        **configs
    )


def SHCullingFullPruningTrainer(
    model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        FullPruningTrainer,
        model, dataset,
        **configs
    )


def SHCullingFullReducedDensificationTrainer(
    model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        FullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


def SHCullingOpacityResetFullReducedDensificationTrainer(
    model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        OpacityResetFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


# Camera Wrapped Trainer

def CameraSHCullingTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingTrainer,
        model, dataset,
        **configs
    )


def CameraFullPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        FullPruningTrainer,
        model, dataset,
        **configs
    )


def CameraFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        FullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


def CameraOpacityResetFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        OpacityResetFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


# Camera and SH Culling Wrapped Trainer

def CameraSHCullingOpacityResetDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingOpacityResetDensificationTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingFullPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingFullPruningTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingOpacityResetFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingOpacityResetFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


# 2DGS normal consistency and depth distortion trainers

NormalFullPruningTrainer = partial(NormalTrainerWrapper, FullPruningTrainer)
NormalFullReducedDensificationTrainer = partial(NormalTrainerWrapper, FullReducedDensificationTrainer)
NormalOpacityResetFullReducedDensificationTrainer = partial(NormalTrainerWrapper, OpacityResetFullReducedDensificationTrainer)

NormalSHCullingOpacityResetDensificationTrainer = partial(NormalTrainerWrapper, SHCullingOpacityResetDensificationTrainer)
NormalSHCullingFullPruningTrainer = partial(NormalTrainerWrapper, SHCullingFullPruningTrainer)
NormalSHCullingFullReducedDensificationTrainer = partial(NormalTrainerWrapper, SHCullingFullReducedDensificationTrainer)
NormalSHCullingOpacityResetFullReducedDensificationTrainer = partial(NormalTrainerWrapper, SHCullingOpacityResetFullReducedDensificationTrainer)

NormalCameraSHCullingTrainer = partial(NormalTrainerWrapper, CameraSHCullingTrainer)
NormalCameraFullPruningTrainer = partial(NormalTrainerWrapper, CameraFullPruningTrainer)
NormalCameraFullReducedDensificationTrainer = partial(NormalTrainerWrapper, CameraFullReducedDensificationTrainer)
NormalCameraOpacityResetFullReducedDensificationTrainer = partial(NormalTrainerWrapper, CameraOpacityResetFullReducedDensificationTrainer)

NormalCameraSHCullingOpacityResetDensificationTrainer = partial(NormalTrainerWrapper, CameraSHCullingOpacityResetDensificationTrainer)
NormalCameraSHCullingFullPruningTrainer = partial(NormalTrainerWrapper, CameraSHCullingFullPruningTrainer)
NormalCameraSHCullingFullReducedDensificationTrainer = partial(NormalTrainerWrapper, CameraSHCullingFullReducedDensificationTrainer)
NormalCameraSHCullingOpacityResetFullReducedDensificationTrainer = partial(NormalTrainerWrapper, CameraSHCullingOpacityResetFullReducedDensificationTrainer)


__all__ = [
    "FullPruningDensifierWrapper",
    "FullPruningTrainerWrapper",
    "BaseFullPruningTrainer",
    "FullPruningTrainer",
    "FullReducedDensificationDensifierWrapper",
    "FullReducedDensificationTrainerWrapper",
    "BaseFullReducedDensificationTrainer",
    "FullReducedDensificationTrainer",
    "OpacityResetFullReducedDensificationTrainer",

    "SHCullingOpacityResetDensificationTrainer",
    "SHCullingFullPruningTrainer",
    "SHCullingFullReducedDensificationTrainer",
    "SHCullingOpacityResetFullReducedDensificationTrainer",

    "CameraSHCullingTrainer",
    "CameraFullPruningTrainer",
    "CameraFullReducedDensificationTrainer",
    "CameraOpacityResetFullReducedDensificationTrainer",
    "CameraSHCullingOpacityResetDensificationTrainer",
    "CameraSHCullingFullPruningTrainer",
    "CameraSHCullingFullReducedDensificationTrainer",
    "CameraSHCullingOpacityResetFullReducedDensificationTrainer",

    "NormalFullPruningTrainer",
    "NormalFullReducedDensificationTrainer",
    "NormalOpacityResetFullReducedDensificationTrainer",
    "NormalSHCullingOpacityResetDensificationTrainer",
    "NormalSHCullingFullPruningTrainer",
    "NormalSHCullingFullReducedDensificationTrainer",
    "NormalSHCullingOpacityResetFullReducedDensificationTrainer",
    "NormalCameraSHCullingTrainer",
    "NormalCameraFullPruningTrainer",
    "NormalCameraFullReducedDensificationTrainer",
    "NormalCameraOpacityResetFullReducedDensificationTrainer",
    "NormalCameraSHCullingOpacityResetDensificationTrainer",
    "NormalCameraSHCullingFullPruningTrainer",
    "NormalCameraSHCullingFullReducedDensificationTrainer",
    "NormalCameraSHCullingOpacityResetFullReducedDensificationTrainer",
]
