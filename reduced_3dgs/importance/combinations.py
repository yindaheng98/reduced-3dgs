from typing import Callable, List
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, DepthTrainerWrapper, NoopDensifier, DensificationTrainerWrapper
from .trainer import ImportancePruningTrainerWrapper, BaseImportancePruningTrainer


def ImportancePrunerInDensifyTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float, List[Camera]], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args,
        importance_prune_from_iter=15000,
        importance_prune_until_iter=20000,
        importance_prune_interval: int = 1000,
        importance_score_resize=None,
        importance_prune_type="comprehensive",
        importance_prune_percent=0.1,
        importance_prune_thr_important_score=None,
        importance_prune_thr_v_important_score=3.0,
        importance_prune_thr_max_v_important_score=None,
        importance_prune_thr_count=1,
        importance_prune_thr_T_alpha=1.0,
        importance_prune_thr_T_alpha_avg=0.001,
        importance_v_pow=0.1,
        **kwargs):
    return DensificationTrainerWrapper(
        lambda model, scene_extent: ImportancePruningTrainerWrapper(
            noargs_base_densifier_constructor,
            model, scene_extent, dataset,
            importance_prune_from_iter=importance_prune_from_iter,
            importance_prune_until_iter=importance_prune_until_iter,
            importance_prune_interval=importance_prune_interval,
            importance_score_resize=importance_score_resize,
            importance_prune_type=importance_prune_type,
            importance_prune_percent=importance_prune_percent,
            importance_prune_thr_important_score=importance_prune_thr_important_score,
            importance_prune_thr_v_important_score=importance_prune_thr_v_important_score,
            importance_prune_thr_max_v_important_score=importance_prune_thr_max_v_important_score,
            importance_prune_thr_count=importance_prune_thr_count,
            importance_prune_thr_T_alpha=importance_prune_thr_T_alpha,
            importance_prune_thr_T_alpha_avg=importance_prune_thr_T_alpha_avg,
            importance_v_pow=importance_v_pow,
        ),
        model, scene_extent,
        *args, **kwargs
    )


def BaseImportancePrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: List[Camera],
        *args, **kwargs):
    return ImportancePrunerInDensifyTrainerWrapper(
        lambda model, scene_extent, dataset: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


# Depth trainer


def DepthImportancePruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseImportancePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


def DepthImportancePrunerInDensifyTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseImportancePrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs)


ImportancePruningTrainer = DepthImportancePruningTrainer
ImportancePrunerInDensifyTrainer = DepthImportancePrunerInDensifyTrainer
