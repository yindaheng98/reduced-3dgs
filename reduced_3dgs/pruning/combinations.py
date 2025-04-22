
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper
from .trainer import BasePruningTrainer, BasePrunerInDensifyTrainer


# Camera trainer


def DepthPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, depth_from_iter=500, **kwargs):
    return DepthTrainerWrapper(BasePruningTrainer, model, scene_extent, *args, dataset=dataset, depth_from_iter=depth_from_iter, **kwargs)


def DepthPrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, depth_from_iter=500, **kwargs):
    return DepthTrainerWrapper(BasePrunerInDensifyTrainer, model, scene_extent, *args, dataset=dataset, depth_from_iter=depth_from_iter, **kwargs)


PruningTrainer = DepthPruningTrainer
PrunerInDensifyTrainer = DepthPrunerInDensifyTrainer
