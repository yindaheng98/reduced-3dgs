from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import NoopDensifier, DensificationTrainer, DensificationDensifierWrapper
from .trainer import ImportancePruningDensifierWrapper, BaseImportancePruningTrainer


def DepthImportancePruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseImportancePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


ImportancePruningTrainer = DepthImportancePruningTrainer


def ImportancePrunerInDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs) -> AbstractDensifier:
    return ImportancePruningDensifierWrapper(
        partial(DensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def ImportancePrunerInDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(ImportancePrunerInDensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseImportancePrunerInDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return ImportancePrunerInDensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthImportancePrunerInDensificationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseImportancePrunerInDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


ImportancePrunerInDensificationTrainer = DepthImportancePrunerInDensificationTrainer
