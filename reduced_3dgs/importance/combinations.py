from typing import List
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, NoopDensifier, DensificationTrainerWrapper
from .trainer import ImportancePruner, BaseImportancePruningTrainer


def BaseImportancePrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=1000,
        importance_prune_until_iter=15000,
        importance_prune_interval=100,
        **kwargs):
    return DensificationTrainerWrapper(
        lambda model, scene_extent: ImportancePruner(
            NoopDensifier(model),
            dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
        ),
        model,
        scene_extent,
        *args, **kwargs
    )


# Depth trainer


def DepthImportancePruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseImportancePruningTrainer, model, scene_extent, *args, dataset=dataset, **kwargs)


def DepthImportancePrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseImportancePrunerInDensifyTrainer, model, scene_extent, *args, dataset=dataset, **kwargs)


ImportancePruningTrainer = DepthImportancePruningTrainer
ImportancePrunerInDensifyTrainer = DepthImportancePrunerInDensifyTrainer
