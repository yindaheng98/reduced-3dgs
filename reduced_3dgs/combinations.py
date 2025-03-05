from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import OpacityResetDensificationTrainer, OpacityResetter, CameraOptimizer
from .shculling import VariableSHGaussianModel, SHCuller, BaseSHCullingTrainer
from .pruning import BasePruningTrainer, BasePrunerInDensifyTrainer


def OpacityResetPrunerInDensifyTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        opacity_reset_interval=3000,
        *args, **kwargs):
    return OpacityResetter(
        BasePrunerInDensifyTrainer(
            model, scene_extent, dataset,
            *args, **kwargs
        ),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )


def SHCullingDensifyTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )


def SHCullingPruneTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        BasePruningTrainer(model, scene_extent, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )


def SHCullingPruningDensifyTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        cdist_threshold: float = 6,
        std_threshold: float = 0.04,
        cull_at_steps=[15000],
        *args, **kwargs):
    return SHCuller(
        OpacityResetPrunerInDensifyTrainer(model, scene_extent, dataset, *args, **kwargs),
        dataset,
        cdist_threshold=cdist_threshold,
        std_threshold=std_threshold,
        cull_at_steps=cull_at_steps,
    )


class CameraTrainableVariableSHGaussianModel(VariableSHGaussianModel):
    def forward(self, camera: Camera):
        return CameraTrainableGaussianModel.forward(self, camera)


def _CameraTrainerWrapper(
    base_trainer_constructor,
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        camera_position_lr_init=0.00016,
        camera_position_lr_final=0.0000016,
        camera_position_lr_delay_mult=0.01,
        camera_position_lr_max_steps=30_000,
        camera_rotation_lr_init=0.0001,
        camera_rotation_lr_final=0.000001,
        camera_rotation_lr_delay_mult=0.01,
        camera_rotation_lr_max_steps=30_000,
        *args, **kwargs):
    return CameraOptimizer(
        base_trainer_constructor(model, scene_extent, dataset, *args, **kwargs),
        dataset, scene_extent,
        camera_position_lr_init=camera_position_lr_init,
        camera_position_lr_final=camera_position_lr_final,
        camera_position_lr_delay_mult=camera_position_lr_delay_mult,
        camera_position_lr_max_steps=camera_position_lr_max_steps,
        camera_rotation_lr_init=camera_rotation_lr_init,
        camera_rotation_lr_final=camera_rotation_lr_final,
        camera_rotation_lr_delay_mult=camera_rotation_lr_delay_mult,
        camera_rotation_lr_max_steps=camera_rotation_lr_max_steps
    )


def CameraSHCullingTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return _CameraTrainerWrapper(
        BaseSHCullingTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return _CameraTrainerWrapper(
        BasePruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraOpacityResetPrunerInDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return _CameraTrainerWrapper(
        OpacityResetPrunerInDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return _CameraTrainerWrapper(
        SHCullingDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingPruneTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return _CameraTrainerWrapper(
        SHCullingPruneTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingPruningDensifyTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return _CameraTrainerWrapper(
        SHCullingPruningDensifyTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
