from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, BaseTrainer
from .abc import QuantizeTrainerWrapper
from .exclude_zeros import ExcludeZeroSHQuantizer


def VectorQuantizeTrainerWrapper(
    base_trainer: AbstractTrainer,
        num_clusters=256,
        num_clusters_rotation_re=None,
        num_clusters_rotation_im=None,
        num_clusters_opacity=None,
        num_clusters_scaling=None,
        num_clusters_features_dc=None,
        num_clusters_features_rest=[],
        quantizate_from_iter=5000,
        quantizate_until_iter=30000,
        quantizate_interval=500,
):
    return QuantizeTrainerWrapper(
        base_trainer, ExcludeZeroSHQuantizer(
            base_trainer.model,
            num_clusters=num_clusters,
            num_clusters_rotation_re=num_clusters_rotation_re,
            num_clusters_rotation_im=num_clusters_rotation_im,
            num_clusters_opacity=num_clusters_opacity,
            num_clusters_scaling=num_clusters_scaling,
            num_clusters_features_dc=num_clusters_features_dc,
            num_clusters_features_rest=num_clusters_features_rest,
        ),
        quantizate_from_iter, quantizate_until_iter, quantizate_interval
    )


def BaseVectorQuantizeTrainer(
    model: GaussianModel,
    spatial_lr_scale: float,
        num_clusters=256,
        num_clusters_rotation_re=None,
        num_clusters_rotation_im=None,
        num_clusters_opacity=None,
        num_clusters_scaling=None,
        num_clusters_features_dc=None,
        num_clusters_features_rest=[],
        quantizate_from_iter=5000,
        quantizate_until_iter=30000,
        quantizate_interval=1000,
        *args, **kwargs):
    return QuantizeTrainerWrapper(
        BaseTrainer(model, spatial_lr_scale, *args, **kwargs),
        ExcludeZeroSHQuantizer(
            model,
            num_clusters=num_clusters,
            num_clusters_rotation_re=num_clusters_rotation_re,
            num_clusters_rotation_im=num_clusters_rotation_im,
            num_clusters_opacity=num_clusters_opacity,
            num_clusters_scaling=num_clusters_scaling,
            num_clusters_features_dc=num_clusters_features_dc,
            num_clusters_features_rest=num_clusters_features_rest,
        ),
        quantizate_from_iter, quantizate_until_iter, quantizate_interval
    )
