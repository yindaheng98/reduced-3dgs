
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper
from .trainer import BasePruningTrainer, BasePrunerInDensifyTrainer


# Depth trainer


def DepthPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BasePruningTrainer, model, scene_extent, *args, dataset=dataset, **kwargs)


def DepthPrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BasePrunerInDensifyTrainer, model, scene_extent, *args, dataset=dataset, **kwargs)


PruningTrainer = DepthPruningTrainer
PrunerInDensifyTrainer = DepthPrunerInDensifyTrainer
