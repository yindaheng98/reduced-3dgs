from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from reduced_3dgs.quantization import VectorQuantizeTrainerWrapper
from reduced_3dgs import SHCullingOpacityResetDensificationTrainer
from reduced_3dgs import FullPruningTrainer, SHCullingFullPruningTrainer
from reduced_3dgs import OpacityResetFullReducedDensificationTrainer, SHCullingOpacityResetFullReducedDensificationTrainer
from reduced_3dgs import CameraSHCullingOpacityResetDensificationTrainer
from reduced_3dgs import CameraFullPruningTrainer, CameraSHCullingFullPruningTrainer
from reduced_3dgs import CameraOpacityResetFullReducedDensificationTrainer, CameraSHCullingOpacityResetFullReducedDensificationTrainer

backends = ["inria", "gsplat", "gsplat-2dgs"]


def get_gaussian_model_class(backend: str, trainable_camera: bool = False) -> Callable[[int], GaussianModel]:
    match backend:
        case "inria":
            from reduced_3dgs.shculling import VariableSHGaussianModel, CameraTrainableVariableSHGaussianModel
            return CameraTrainableVariableSHGaussianModel if trainable_camera else VariableSHGaussianModel
        case "gsplat":
            from reduced_3dgs.shculling import VariableSHGsplatGaussianModel, CameraTrainableVariableSHGsplatGaussianModel
            return CameraTrainableVariableSHGsplatGaussianModel if trainable_camera else VariableSHGsplatGaussianModel
        case "gsplat-2dgs":
            from reduced_3dgs.shculling import VariableSHGsplat2DGSGaussianModel, CameraTrainableVariableSHGsplat2DGSGaussianModel
            return CameraTrainableVariableSHGsplat2DGSGaussianModel if trainable_camera else VariableSHGsplat2DGSGaussianModel
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def prepare_gaussians(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None, backend: str = "inria") -> GaussianModel:
    gaussians = get_gaussian_model_class(backend, trainable_camera=trainable_camera)(sh_degree).to(device)
    gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians


modes = {
    "densify-shculling": SHCullingOpacityResetDensificationTrainer,
    "pruning": FullPruningTrainer,
    "pruning-shculling": SHCullingFullPruningTrainer,
    "densify-pruning": OpacityResetFullReducedDensificationTrainer,
    "densify-pruning-shculling": SHCullingOpacityResetFullReducedDensificationTrainer,
    "camera-densify-shculling": CameraSHCullingOpacityResetDensificationTrainer,
    "camera-pruning": CameraFullPruningTrainer,
    "camera-pruning-shculling": CameraSHCullingFullPruningTrainer,
    "camera-densify-pruning": CameraOpacityResetFullReducedDensificationTrainer,
    "camera-densify-pruning-shculling": CameraSHCullingOpacityResetFullReducedDensificationTrainer,
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
