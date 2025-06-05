from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from reduced_3dgs.quantization import VectorQuantizeTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainer
from reduced_3dgs.pruning import PruningTrainer
from reduced_3dgs.combinations import PrunerInDensifyTrainer, SHCullingDensificationTrainer, SHCullingPruningTrainer, SHCullingPrunerInDensifyTrainer
from reduced_3dgs.combinations import CameraTrainableVariableSHGaussianModel, CameraSHCullingTrainer, CameraPruningTrainer
from reduced_3dgs.combinations import CameraPrunerInDensifyTrainer, CameraSHCullingDensifyTrainer, CameraSHCullingPruningTrainer, CameraSHCullingPruningDensifyTrainer


def prepare_gaussians(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None) -> GaussianModel:
    if trainable_camera:
        gaussians = CameraTrainableVariableSHGaussianModel(sh_degree).to(device)
        gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    else:
        gaussians = VariableSHGaussianModel(sh_degree).to(device)
        gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians


modes = {
    "shculling": SHCullingTrainer,
    "pruning": PruningTrainer,
    "densify-pruning": PrunerInDensifyTrainer,
    "densify-shculling": SHCullingDensificationTrainer,
    "prune-shculling": SHCullingPruningTrainer,
    "densify-prune-shculling": SHCullingPrunerInDensifyTrainer,
    "camera-shculling": CameraSHCullingTrainer,
    "camera-pruning": CameraPruningTrainer,
    "camera-densify-pruning": CameraPrunerInDensifyTrainer,
    "camera-densify-shculling": CameraSHCullingDensifyTrainer,
    "camera-prune-shculling": CameraSHCullingPruningTrainer,
    "camera-densify-prune-shculling": CameraSHCullingPruningDensifyTrainer,
}


def prepare_quantizer(
        gaussians: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        base_constructor,
        load_quantized: str = None,

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
        **configs):
    trainer = VectorQuantizeTrainerWrapper(
        base_constructor(
            gaussians,
            scene_extent=scene_extent,
            dataset=dataset,
            **configs
        ),

        num_clusters=num_clusters,
        num_clusters_rotation_re=num_clusters_rotation_re,
        num_clusters_rotation_im=num_clusters_rotation_im,
        num_clusters_opacity=num_clusters_opacity,
        num_clusters_scaling=num_clusters_scaling,
        num_clusters_features_dc=num_clusters_features_dc,
        num_clusters_features_rest=num_clusters_features_rest,

        quantize_from_iter=quantize_from_iter,
        quantize_until_iter=quantize_until_iter,
        quantize_interval=quantize_interval,
    )
    if load_quantized:
        trainer.quantizer.load_quantized(load_quantized)
    return trainer, trainer.quantizer


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, with_scale_reg=False, quantize: bool = False, load_quantized: str = None, configs={}) -> AbstractTrainer:
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **kwargs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **kwargs)
    if quantize:
        trainer, quantizer = prepare_quantizer(
            gaussians,
            scene_extent=dataset.scene_extent(),
            dataset=dataset,
            base_constructor=modes[mode],
            load_quantized=load_quantized,
            **configs
        )
    else:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            dataset=dataset,
            **configs
        )
        quantizer = None
    return trainer, quantizer
