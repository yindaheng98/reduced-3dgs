from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import NoopDensifier, DensificationTrainer, DensificationDensifierWrapper
from .trainer import ImportancePruningDensifierWrapper, BaseImportancePruningTrainer


def DepthImportancePruningTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseImportancePruningTrainer,
        model, dataset,
        **configs)


ImportancePruningTrainer = DepthImportancePruningTrainer


def ImportancePrunerInDensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs) -> AbstractDensifier:
    return ImportancePruningDensifierWrapper(
        partial(DensificationDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def ImportancePrunerInDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(ImportancePrunerInDensificationDensifierWrapper, base_densifier_constructor),
        model, dataset,
        **configs
    )


def BaseImportancePrunerInDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return ImportancePrunerInDensificationTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthImportancePrunerInDensificationTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseImportancePrunerInDensificationTrainer,
        model, dataset,
        **configs)


ImportancePrunerInDensificationTrainer = DepthImportancePrunerInDensificationTrainer
