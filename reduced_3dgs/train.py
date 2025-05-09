import os
import random
import shutil
from typing import List, Tuple
import torch
from tqdm import tqdm
from argparse import Namespace
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, JSONCameraDataset, TrainableCameraDataset
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset.colmap import ColmapCameraDataset, ColmapTrainableCameraDataset, colmap_init
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from reduced_3dgs.quantization import AbstractQuantizer, VectorQuantizeTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainer
from reduced_3dgs.pruning import PruningTrainer
from reduced_3dgs.combinations import PrunerInDensifyTrainer, SHCullingDensificationTrainer, SHCullingPruningTrainer, SHCullingPrunerInDensifyTrainer
from reduced_3dgs.combinations import CameraTrainableVariableSHGaussianModel, CameraSHCullingTrainer, CameraPruningTrainer
from reduced_3dgs.combinations import CameraPrunerInDensifyTrainer, CameraSHCullingDensifyTrainer, CameraSHCullingPruningTrainer, CameraSHCullingPruningDensifyTrainer


basemodes = {
    "shculling": SHCullingTrainer,
    "pruning": PruningTrainer,
    "densify-pruning": PrunerInDensifyTrainer,
    "densify-shculling": SHCullingDensificationTrainer,
    "prune-shculling": SHCullingPruningTrainer,
    "densify-prune-shculling": SHCullingPrunerInDensifyTrainer,
}
cameramodes = {
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


def prepare_training(sh_degree: int, source: str, device: str, mode: str, load_ply: str = None, load_camera: str = None, load_depth=False, with_scale_reg=False, quantize: bool = False, load_quantized: str = None, configs={}) -> Tuple[CameraDataset, GaussianModel, AbstractTrainer]:
    quantizer = None
    if mode in basemodes:
        gaussians = VariableSHGaussianModel(sh_degree).to(device)
        gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
        dataset = (JSONCameraDataset(load_camera, load_depth=load_depth) if load_camera else ColmapCameraDataset(source, load_depth=load_depth)).to(device)
        modes = basemodes
    elif mode in cameramodes:
        gaussians = CameraTrainableVariableSHGaussianModel(sh_degree).to(device)
        gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
        dataset = (TrainableCameraDataset.from_json(load_camera, load_depth=load_depth) if load_camera else ColmapTrainableCameraDataset(source, load_depth=load_depth)).to(device)
        modes = cameramodes
    else:
        raise ValueError(f"Unknown mode: {mode}")
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
    return dataset, gaussians, trainer, quantizer


def save_cfg_args(destination: str, sh_degree: int, source: str):
    os.makedirs(destination, exist_ok=True)
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))


def training(dataset: CameraDataset, gaussians: GaussianModel, trainer: AbstractTrainer, quantizer: AbstractQuantizer, destination: str, iteration: int, save_iterations: List[int], device: str, empty_cache_every_step=False):
    shutil.rmtree(os.path.join(destination, "point_cloud"), ignore_errors=True)  # remove the previous point cloud
    pbar = tqdm(range(1, iteration+1))
    epoch = list(range(len(dataset)))
    epoch_psnr = torch.empty(3, 0, device=device)
    ema_loss_for_log = 0.0
    avg_psnr_for_log = 0.0
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            avg_psnr_for_log = epoch_psnr.mean().item()
            epoch_psnr = torch.empty(3, 0, device=device)
            random.shuffle(epoch)
        idx = epoch[epoch_idx]
        loss, out = trainer.step(dataset[idx])
        if empty_cache_every_step:
            torch.cuda.empty_cache()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            epoch_psnr = torch.concat([epoch_psnr, psnr(out["render"], dataset[idx].ground_truth_image)], dim=1)
            if step % 10 == 0:
                pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'n': gaussians._xyz.shape[0]})
        if step in save_iterations:
            save_path = os.path.join(destination, "point_cloud", "iteration_" + str(step))
            os.makedirs(save_path, exist_ok=True)
            gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
            dataset.save_cameras(os.path.join(destination, "cameras.json"))
            if quantizer:
                quantizer.save_quantized(gaussians, os.path.join(save_path, "point_cloud_quantized.ply"))
    save_path = os.path.join(destination, "point_cloud", "iteration_" + str(iteration))
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    dataset.save_cameras(os.path.join(destination, "cameras.json"))
    if quantizer:
        quantizer.save_quantized(gaussians, os.path.join(save_path, "point_cloud_quantized.ply"))


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("-l", "--load_ply", default=None, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--no_depth_data", action='store_true')
    parser.add_argument("--with_scale_reg", action="store_true")
    parser.add_argument("--load_quantized", default=None, type=str)
    parser.add_argument("--mode", choices=list(basemodes.keys()) + list(cameramodes.keys()), default="densify-prune-shculling")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--empty_cache_every_step", action='store_true')
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, gaussians, trainer, quantizer = prepare_training(
        sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode,
        load_ply=args.load_ply, load_camera=args.load_camera, load_depth=not args.no_depth_data, with_scale_reg=args.with_scale_reg,
        quantize=args.quantize, load_quantized=args.load_quantized, configs=configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    torch.cuda.empty_cache()
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer, quantizer=quantizer,
        destination=args.destination, iteration=args.iteration, save_iterations=args.save_iterations,
        device=args.device)
