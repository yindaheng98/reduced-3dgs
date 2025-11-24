
from typing import Callable, List
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper, NoopDensifier, SplitCloneDensifierTrainerWrapper
from .trainer import BasePruner, BasePruningTrainer


def PrunerInDensifyTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float, List[Camera]], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval: int = 100,
        box_size=1.,
        lambda_mercy=1.,
        mercy_minimum=3,
        mercy_type='redundancy_opacity',
        **kwargs):
    return SplitCloneDensifierTrainerWrapper(
        lambda model, scene_extent: BasePruner(
            noargs_base_densifier_constructor(model, scene_extent, dataset),
            scene_extent, dataset,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            box_size=box_size,
            lambda_mercy=lambda_mercy,
            mercy_minimum=mercy_minimum,
            mercy_type=mercy_type,
        ),
        model,
        scene_extent,
        *args, **kwargs
    )


def BasePrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args, **kwargs):
    return PrunerInDensifyTrainerWrapper(
        lambda model, scene_extent, dataset: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


# Depth trainer


def DepthPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BasePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


def DepthPrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BasePrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


PruningTrainer = DepthPruningTrainer
PrunerInDensifyTrainer = DepthPrunerInDensifyTrainer
