import os
from typing import Dict
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from gaussian_splatting import GaussianModel
from plyfile import PlyData, PlyElement
import numpy as np
from gaussian_splatting.trainer import AbstractTrainer, BaseTrainer
from .abc import AbstractQuantizer, QuantizeTrainerWrapper


def generate_codebook(values: torch.Tensor, num_clusters=256, tol=0.0001, max_iter=500, init_codebook=None):
    kmeans = KMeans(
        n_clusters=num_clusters, tol=tol, max_iter=max_iter,
        init='random' if init_codebook is None else init_codebook.cpu().numpy(),
        random_state=0, n_init="auto", verbose=0,
        batch_size=256 * os.cpu_count()
    )
    ids = torch.tensor(kmeans.fit_predict(values.cpu().numpy()), device=values.device)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=values.dtype, device=values.device)
    return centers, ids


def produce_clusters(self: GaussianModel, num_clusters: int, init_codebook_dict={}):
    codebook_dict: Dict[str, torch.Tensor] = {}
    ids_dict: Dict[str, torch.Tensor] = {}

    init_codebook_dict = {
        "features_dc": None,
        "features_dc": None,
        **{f"features_rest_{sh_degree}": None for sh_degree in range(self.max_sh_degree)},
        "rotation_re": None,
        "rotation_im": None,
        "opacity": None,
        "scaling": None,
        **init_codebook_dict
    }

    codebook_dict["features_dc"], ids = generate_codebook(self._features_dc.detach().squeeze(1), num_clusters=num_clusters, tol=0.001, init_codebook=init_codebook_dict["features_dc"])
    ids_dict["features_dc"] = ids.unsqueeze(1)
    features_rest_flatten = self._features_rest.detach().transpose(1, 2).flatten(0, 1)
    for sh_degree in range(self.max_sh_degree):
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        codebook_dict[f"features_rest_{sh_degree}"], ids = generate_codebook(features_rest_flatten[:, sh_idx_start:sh_idx_end], num_clusters=num_clusters, init_codebook=init_codebook_dict[f"features_rest_{sh_degree}"])
        ids_dict[f"features_rest_{sh_degree}"] = ids.reshape(-1, self._features_rest.shape[-1])

    codebook_dict["rotation_re"], ids_dict[f"rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1], num_clusters=num_clusters, init_codebook=init_codebook_dict["rotation_re"])
    codebook_dict["rotation_im"], ids_dict[f"rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:], num_clusters=num_clusters, init_codebook=init_codebook_dict["rotation_im"])

    codebook_dict["opacity"], ids_dict[f"opacity"] = generate_codebook(self._opacity.detach(), num_clusters=num_clusters, init_codebook=init_codebook_dict["opacity"])
    codebook_dict["scaling"], ids_dict[f"scaling"] = generate_codebook(self._scaling.detach(), num_clusters=num_clusters, init_codebook=init_codebook_dict["scaling"])
    return codebook_dict, ids_dict


def apply_clustering(self: GaussianModel, codebook_dict: Dict[str, torch.Tensor], ids_dict: Dict[str, torch.Tensor]):

    opacity = codebook_dict["opacity"][ids_dict["opacity"], ...]
    scaling = codebook_dict["scaling"][ids_dict["scaling"], ...]

    rotation = torch.cat((
        codebook_dict["rotation_re"][ids_dict["rotation_re"], ...],
        codebook_dict["rotation_im"][ids_dict["rotation_im"], ...],
    ), dim=1)

    features_rest = []
    for sh_degree in range(self.max_sh_degree):
        features_rest.append(codebook_dict[f"features_rest_{sh_degree}"][ids_dict[f"features_rest_{sh_degree}"], ...])
    features_rest = torch.cat(features_rest, dim=2).transpose(1, 2)

    features_dc = codebook_dict["features_dc"][ids_dict["features_dc"], ...]

    with torch.no_grad():
        self._opacity[...] = opacity
        self._scaling[...] = scaling
        self._rotation[...] = rotation
        self._features_dc[...] = features_dc
        self._features_rest[...] = features_rest
    return self


class VectorQuantizer(AbstractQuantizer):
    def __init__(self, model: GaussianModel, num_clusters=256):
        self._model = model
        self.num_clusters = num_clusters
        self._codebook_dict = {}

    @property
    def model(self) -> GaussianModel:
        return self._model

    def quantize(self) -> GaussianModel:
        model = self.model
        codebook_dict, ids_dict = produce_clusters(model, self.num_clusters, self._codebook_dict)
        self._codebook_dict = codebook_dict
        return apply_clustering(model, codebook_dict, ids_dict)

    def save_quantized(self, ply_path: str):
        model = self.model
        codebook_dict, ids_dict = produce_clusters(model, self.num_clusters, self._codebook_dict)
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('scale', 'u4'),
            ('rot_re', 'u4'),
            ('rot_im', 'u4'),
            ('opacity', 'u4'),
            ('f_dc', 'u4'),
        ]
        for sh_degree in range(model.max_sh_degree):
            dtype_full.extend([
                (f'f_rest_{sh_degree}_0', 'u4'),
                (f'f_rest_{sh_degree}_1', 'u4'),
                (f'f_rest_{sh_degree}_2', 'u4'),
            ])
        data_full = [
            *np.array_split(model._xyz.detach().cpu().numpy(), 3, axis=1),
            *np.array_split(torch.zeros_like(model._xyz).detach().cpu().numpy(), 3, axis=1),
            ids_dict["scaling"].unsqueeze(-1).cpu().numpy(),
            ids_dict["rotation_re"].unsqueeze(-1).cpu().numpy(),
            ids_dict["rotation_im"].unsqueeze(-1).cpu().numpy(),
            ids_dict["opacity"].unsqueeze(-1).cpu().numpy(),
            ids_dict["features_dc"].cpu().numpy(),
        ]
        for sh_degree in range(model.max_sh_degree):
            features_rest = ids_dict[f'features_rest_{sh_degree}'].cpu().numpy()
            data_full.extend(np.array_split(features_rest, 3, axis=1))

        elements = np.rec.fromarrays([data.squeeze(-1) for data in data_full], dtype=dtype_full)
        el = PlyElement.describe(elements, 'vertex')

        dtype_full = [
            ("scaling_0", 'f4'), ("scaling_1", 'f4'), ("scaling_2", 'f4'),
            ("rot_re", 'f4'), ("rot_im_0", 'f4'), ("rot_im_1", 'f4'), ("rot_im_2", 'f4'),
            ("opacity", 'f4'),
            ("f_dc_0", 'f4'), ("f_dc_1", 'f4'), ("f_dc_2", 'f4'),
        ]
        for sh_degree in range(model.max_sh_degree):
            features_rest = codebook_dict[f'features_rest_{sh_degree}']
            n_channels = (sh_degree + 2) ** 2 - (sh_degree + 1) ** 2
            dtype_full.extend([(f'f_rest_{sh_degree}_{ch}', 'f4') for ch in range(n_channels)])
        data_full = [
            *np.array_split(codebook_dict["scaling"].cpu().numpy(), 3, axis=1),
            codebook_dict["rotation_re"].cpu().numpy(),
            *np.array_split(codebook_dict["rotation_im"].cpu().numpy(), 3, axis=1),
            codebook_dict["opacity"].cpu().numpy(),
            *np.array_split(codebook_dict["features_dc"].cpu().numpy(), 3, axis=1),
        ]
        for sh_degree in range(model.max_sh_degree):
            features_rest = codebook_dict[f'features_rest_{sh_degree}'].cpu().numpy()
            n_channels = (sh_degree + 2) ** 2 - (sh_degree + 1) ** 2
            data_full.extend(np.array_split(features_rest, n_channels, axis=1))

        codebook = np.rec.fromarrays([data.squeeze(-1) for data in data_full], dtype=dtype_full)
        cb = PlyElement.describe(codebook, 'codebook')

        PlyData([el, cb]).write(ply_path)

        return apply_clustering(model, codebook_dict, ids_dict)

    def load_quantized(self, ply_path: str):
        model = self.model
        plydata = PlyData.read(ply_path)

        ids_dict = {}
        elements = plydata['vertex']
        kwargs = dict(dtype=torch.long, device=model._xyz.device)
        ids_dict["scaling"] = torch.tensor(elements["scale"], **kwargs)
        ids_dict["rotation_re"] = torch.tensor(elements["rot_re"], **kwargs)
        ids_dict["rotation_im"] = torch.tensor(elements["rot_im"], **kwargs)
        ids_dict["opacity"] = torch.tensor(elements["opacity"], **kwargs)
        ids_dict["features_dc"] = torch.tensor(elements["f_dc"], **kwargs).unsqueeze(-1)
        for sh_degree in range(model.max_sh_degree):
            ids_dict[f'features_rest_{sh_degree}'] = torch.tensor(np.stack([elements[f'f_rest_{sh_degree}_{ch}'] for ch in range(3)], axis=1), **kwargs)

        codebook_dict = {}
        codebook = plydata['codebook']
        kwargs = dict(dtype=torch.float32, device=model._xyz.device)
        codebook_dict["scaling"] = torch.tensor(np.stack([codebook[f'scaling_{ch}'] for ch in range(3)], axis=1), **kwargs)
        codebook_dict["rotation_re"] = torch.tensor(codebook["rot_re"], **kwargs).unsqueeze(-1)
        codebook_dict["rotation_im"] = torch.tensor(np.stack([codebook[f'rot_im_{ch}'] for ch in range(3)], axis=1), **kwargs)
        codebook_dict["opacity"] = torch.tensor(codebook["opacity"], **kwargs).unsqueeze(-1)
        codebook_dict["features_dc"] = torch.tensor(np.stack([codebook[f'f_dc_{ch}'] for ch in range(3)], axis=1), **kwargs)
        for sh_degree in range(model.max_sh_degree):
            n_channels = (sh_degree + 2) ** 2 - (sh_degree + 1) ** 2
            codebook_dict[f'features_rest_{sh_degree}'] = torch.tensor(np.stack([codebook[f'f_rest_{sh_degree}_{ch}'] for ch in range(n_channels)], axis=1), **kwargs)

        return apply_clustering(model, codebook_dict, ids_dict)


def VectorQuantizeTrainerWrapper(
        base_trainer: AbstractTrainer,
        num_clusters=256,
        quantizate_from_iter=5000,
        quantizate_until_iter=30000,
        quantizate_interval=500,
):
    return QuantizeTrainerWrapper(
        base_trainer, VectorQuantizer(base_trainer.model, num_clusters=num_clusters),
        quantizate_from_iter, quantizate_until_iter, quantizate_interval
    )


def VectorQuantizeTrainer(
    model: GaussianModel,
    spatial_lr_scale: float,
        num_clusters=256,
        quantizate_from_iter=5000,
        quantizate_until_iter=30000,
        quantizate_interval=1000,
        *args, **kwargs):
    return QuantizeTrainerWrapper(
        BaseTrainer(model, spatial_lr_scale, *args, **kwargs),
        VectorQuantizer(model, num_clusters=num_clusters),
        quantizate_from_iter, quantizate_until_iter, quantizate_interval
    )
