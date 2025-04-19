from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, Trainer
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
        quantize_from_iter=5000,
        quantize_until_iter=30000,
        quantize_interval=1000,
        treat_as_zero=1e-8,
):
    return QuantizeTrainerWrapper(
        base_trainer, ExcludeZeroSHQuantizer(
            num_clusters=num_clusters,
            num_clusters_rotation_re=num_clusters_rotation_re,
            num_clusters_rotation_im=num_clusters_rotation_im,
            num_clusters_opacity=num_clusters_opacity,
            num_clusters_scaling=num_clusters_scaling,
            num_clusters_features_dc=num_clusters_features_dc,
            num_clusters_features_rest=num_clusters_features_rest,
            treat_as_zero=treat_as_zero,
        ),
        quantize_from_iter=quantize_from_iter,
        quantize_until_iter=quantize_until_iter,
        quantize_interval=quantize_interval,

    )


def VectorQuantizeTrainer(
    model: GaussianModel,
    scene_extent: float,
        num_clusters=256,
        num_clusters_rotation_re=None,
        num_clusters_rotation_im=None,
        num_clusters_opacity=None,
        num_clusters_scaling=None,
        num_clusters_features_dc=None,
        num_clusters_features_rest=[],
        quantize_from_iter=5000,
        quantize_until_iter=30000,
        quantize_interval=1000,
        treat_as_zero=1e-8,
        *args, **kwargs):
    return VectorQuantizeTrainerWrapper(
        Trainer(model, scene_extent, *args, **kwargs),
        num_clusters=num_clusters,
        num_clusters_rotation_re=num_clusters_rotation_re,
        num_clusters_rotation_im=num_clusters_rotation_im,
        num_clusters_opacity=num_clusters_opacity,
        num_clusters_scaling=num_clusters_scaling,
        num_clusters_features_dc=num_clusters_features_dc,
        num_clusters_features_rest=num_clusters_features_rest,
        treat_as_zero=treat_as_zero,
        quantize_from_iter=quantize_from_iter,
        quantize_until_iter=quantize_until_iter,
        quantize_interval=quantize_interval,
    )
