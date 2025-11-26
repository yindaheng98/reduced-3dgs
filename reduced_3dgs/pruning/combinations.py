
from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import NoopDensifier, DensificationTrainer, SplitCloneDensifierWrapper
from .trainer import PruningDensifierWrapper, BasePruningTrainer


def DepthPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BasePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


PruningTrainer = DepthPruningTrainer


def ReducedDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs) -> AbstractDensifier:
    return PruningDensifierWrapper(
        partial(SplitCloneDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def ReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(ReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseReducedDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return ReducedDensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthReducedDensificationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


ReducedDensificationTrainer = DepthReducedDensificationTrainer
