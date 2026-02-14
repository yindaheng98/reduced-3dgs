
from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import NoopDensifier, DensificationTrainer, SplitCloneDensifierWrapper
from .trainer import PruningDensifierWrapper, BasePruningTrainer


def DepthPruningTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BasePruningTrainer,
        model, dataset,
        **configs)


PruningTrainer = DepthPruningTrainer


def ReducedDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs) -> AbstractDensifier:
    return PruningDensifierWrapper(
        partial(SplitCloneDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def ReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(ReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def BaseReducedDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return ReducedDensificationTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthReducedDensificationTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseReducedDensificationTrainer,
        model, dataset,
        **configs)


ReducedDensificationTrainer = DepthReducedDensificationTrainer
