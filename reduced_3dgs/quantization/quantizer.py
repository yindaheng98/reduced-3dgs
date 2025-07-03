import math
import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
try:
    from cuml.cluster import KMeans
    kmeans_init = 'k-means||'
except ImportError:
    print("Cuml not found, using sklearn's MiniBatchKMeans for quantization.")
    from sklearn.cluster import MiniBatchKMeans
    from functools import partial
    KMeans = partial(MiniBatchKMeans, batch_size=256 * os.cpu_count())
    kmeans_init = 'k-means++'
from gaussian_splatting import GaussianModel
from plyfile import PlyData, PlyElement
import numpy as np
from .abc import AbstractQuantizer


def array2record(array: torch.Tensor, perfix, n_cols, dtype):
    dtype_full = [(f'{perfix}_{i}', dtype) for i in range(n_cols)] if n_cols > 1 else [(perfix, dtype)]
    data_full = map(lambda x: x.squeeze(-1), np.array_split(array.cpu().numpy(), n_cols, axis=1))
    record = np.rec.fromarrays(data_full, dtype=dtype_full)
    return record


def compute_uint_length(n):
    count = 0
    while n >> 1:
        count += 1
        n >>= 1
    return count


def compute_uint_dtype(n):
    bits = compute_uint_length(n)
    bytes = bits // 8
    if bits % 8:
        bytes += 1
    return f'u{bytes}'


class VectorQuantizer(AbstractQuantizer):
    def __init__(
            self,
            num_clusters=256,
            num_clusters_rotation_re=None,
            num_clusters_rotation_im=None,
            num_clusters_opacity=None,
            num_clusters_scaling=None,
            num_clusters_features_dc=None,
            num_clusters_features_rest=[],
            max_sh_degree=3,
            force_code_dtype=None,
            force_codebook_dtype='f4',
            tol=1e-6, max_iter=500,
    ):
        self.num_clusters_rotation_re = num_clusters_rotation_re or num_clusters
        self.num_clusters_rotation_im = num_clusters_rotation_im or num_clusters
        self.num_clusters_opacity = num_clusters_opacity or num_clusters
        self.num_clusters_scaling = num_clusters_scaling or num_clusters
        self.num_clusters_features_dc = num_clusters_features_dc or num_clusters
        self.num_clusters_features_rest = [(num_clusters_features_rest[i] if len(num_clusters_features_rest) > i else num_clusters) for i in range(max_sh_degree)]
        self.force_code_dtype = force_code_dtype
        self.force_codebook_dtype = force_codebook_dtype
        self.tol = tol
        self.max_iter = max_iter

        self._codebook_dict = {}

    def generate_codebook(self, values: torch.Tensor, num_clusters, init_codebook=None):
        kmeans = KMeans(
            n_clusters=num_clusters, tol=self.tol, max_iter=self.max_iter,
            init=kmeans_init if init_codebook is None else init_codebook.cpu().numpy(),
            random_state=0, n_init="auto", verbose=1,
        )
        ids = torch.tensor(kmeans.fit_predict(values.cpu().numpy()), device=values.device)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=values.dtype, device=values.device)
        return centers, ids

    def one_nearst(self, points: torch.Tensor, codebook: torch.Tensor, batch=2**16):
        ids = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
        for i in range(math.ceil(points.shape[0]/batch)):
            ids[i*batch:i*batch+batch] = torch.argmin(torch.cdist(points[i*batch:i*batch+batch, ...], codebook), dim=1)
        return ids

    def produce_clusters_features_dc(self, model: GaussianModel, *args, **kwargs):
        codebook, ids = self.generate_codebook(model._features_dc.detach().squeeze(1), self.num_clusters_features_dc, *args, **kwargs)
        return codebook, ids.unsqueeze(1)

    def find_nearest_cluster_id_features_dc(self, model: GaussianModel, codebook: torch.Tensor):
        return self.one_nearst(model._features_dc.detach().squeeze(1), codebook).unsqueeze(1)

    def produce_clusters_degree_features_rest(self, model: GaussianModel, sh_degree, *args, **kwargs):
        features_rest_flatten = model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        codebook, ids = self.generate_codebook(features_rest, self.num_clusters_features_rest[sh_degree], *args, **kwargs)
        return codebook, ids.reshape(-1, model._features_rest.shape[-1])

    def find_nearest_cluster_id_degree_features_rest(self, model: GaussianModel, sh_degree, codebook: torch.Tensor):
        features_rest_flatten = model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        ids = self.one_nearst(features_rest, codebook)
        return ids.reshape(-1, model._features_rest.shape[-1])

    def produce_clusters_rotation_re(self, model: GaussianModel, *args, **kwargs):
        return self.generate_codebook(model.get_rotation.detach()[:, 0:1], self.num_clusters_rotation_re, *args, **kwargs)

    def find_nearest_cluster_id_rotation_re(self, model: GaussianModel, codebook: torch.Tensor):
        return self.one_nearst(model.get_rotation.detach()[:, 0:1], codebook)

    def produce_clusters_rotation_im(self, model: GaussianModel, *args, **kwargs):
        return self.generate_codebook(model.get_rotation.detach()[:, 1:], self.num_clusters_rotation_im, *args, **kwargs)

    def find_nearest_cluster_id_rotation_im(self, model: GaussianModel, codebook: torch.Tensor):
        return self.one_nearst(model.get_rotation.detach()[:, 1:], codebook)

    def produce_clusters_opacity(self, model: GaussianModel, *args, **kwargs):
        return self.generate_codebook(model._opacity.detach(), self.num_clusters_opacity, *args, **kwargs)

    def find_nearest_cluster_id_opacity(self, model: GaussianModel, codebook: torch.Tensor):
        return self.one_nearst(model._opacity.detach(), codebook)

    def produce_clusters_scaling(self, model: GaussianModel, *args, **kwargs):
        centers, ids = self.generate_codebook(model.get_scaling.detach(), self.num_clusters_scaling, *args, **kwargs)
        centers_log = model.scaling_inverse_activation(centers)
        return centers_log, ids

    def find_nearest_cluster_id_scaling(self, model: GaussianModel, codebook: torch.Tensor):
        return self.one_nearst(model.get_scaling.detach(), model.scaling_activation(codebook))

    def produce_clusters(self, model: GaussianModel, init_codebook_dict={}):
        codebook_dict: Dict[str, torch.Tensor] = {}
        ids_dict: Dict[str, torch.Tensor] = {}
        init_codebook_dict = {
            "features_dc": None,
            **{f"features_rest_{sh_degree}": None for sh_degree in range(model.max_sh_degree)},
            "rotation_re": None,
            "rotation_im": None,
            "opacity": None,
            "scaling": None,
            **init_codebook_dict
        }

        codebook_dict["features_dc"], ids_dict["features_dc"] = self.produce_clusters_features_dc(model, init_codebook=init_codebook_dict["features_dc"])
        for sh_degree in range(model.max_sh_degree):
            codebook_dict[f"features_rest_{sh_degree}"], ids_dict[f"features_rest_{sh_degree}"] = self.produce_clusters_degree_features_rest(
                model, sh_degree, init_codebook=init_codebook_dict[f"features_rest_{sh_degree}"]
            )
        codebook_dict["rotation_re"], ids_dict[f"rotation_re"] = self.produce_clusters_rotation_re(model, init_codebook=init_codebook_dict["rotation_re"])
        codebook_dict["rotation_im"], ids_dict[f"rotation_im"] = self.produce_clusters_rotation_im(model, init_codebook=init_codebook_dict["rotation_im"])
        codebook_dict["opacity"], ids_dict[f"opacity"] = self.produce_clusters_opacity(model, init_codebook=init_codebook_dict["opacity"])
        codebook_dict["scaling"], ids_dict[f"scaling"] = self.produce_clusters_scaling(model, init_codebook=init_codebook_dict["scaling"])
        return codebook_dict, ids_dict

    def find_nearest_cluster_id(self, model: GaussianModel, codebook_dict={}):
        ids_dict: Dict[str, torch.Tensor] = {}
        ids_dict["features_dc"] = self.find_nearest_cluster_id_features_dc(model, codebook=codebook_dict["features_dc"])
        for sh_degree in range(model.max_sh_degree):
            ids_dict[f"features_rest_{sh_degree}"] = self.find_nearest_cluster_id_degree_features_rest(
                model, sh_degree, codebook=codebook_dict[f"features_rest_{sh_degree}"]
            )
        ids_dict[f"rotation_re"] = self.find_nearest_cluster_id_rotation_re(model, codebook=codebook_dict["rotation_re"])
        ids_dict[f"rotation_im"] = self.find_nearest_cluster_id_rotation_im(model, codebook=codebook_dict["rotation_im"])
        ids_dict[f"opacity"] = self.find_nearest_cluster_id_opacity(model, codebook=codebook_dict["opacity"])
        ids_dict[f"scaling"] = self.find_nearest_cluster_id_scaling(model, codebook=codebook_dict["scaling"])
        return ids_dict

    def dequantize(self, model: GaussianModel, ids_dict: Dict[str, torch.Tensor], codebook_dict: Dict[str, torch.Tensor], xyz: torch.Tensor = None, replace=False) -> GaussianModel:
        opacity = codebook_dict["opacity"][ids_dict["opacity"], ...]
        scaling = codebook_dict["scaling"][ids_dict["scaling"], ...]

        rotation = torch.cat((
            codebook_dict["rotation_re"][ids_dict["rotation_re"], ...],
            codebook_dict["rotation_im"][ids_dict["rotation_im"], ...],
        ), dim=1)

        features_dc = codebook_dict["features_dc"][ids_dict["features_dc"], ...]
        features_rest = []
        for sh_degree in range(model.max_sh_degree):
            features_rest.append(codebook_dict[f"features_rest_{sh_degree}"][ids_dict[f"features_rest_{sh_degree}"], ...])
        features_rest = torch.cat(features_rest, dim=2).transpose(1, 2)

        with torch.no_grad():
            if replace:
                if xyz is not None:
                    model._xyz = nn.Parameter(xyz)
                model._opacity = nn.Parameter(opacity)
                model._scaling = nn.Parameter(scaling)
                model._rotation = nn.Parameter(rotation)
                model._features_dc = nn.Parameter(features_dc)
                model._features_rest = nn.Parameter(features_rest)
            else:
                if xyz is not None:
                    model._xyz[...] = xyz
                model._opacity[...] = opacity
                model._scaling[...] = scaling
                model._rotation[...] = rotation
                model._features_dc[...] = features_dc
                model._features_rest[...] = features_rest
        return model

    def quantize(self, model: GaussianModel, update_codebook=True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self._codebook_dict == {} or update_codebook:
            codebook_dict, ids_dict = self.produce_clusters(model, self._codebook_dict)
            self._codebook_dict = codebook_dict
        else:
            codebook_dict = self._codebook_dict
            ids_dict = self.find_nearest_cluster_id(model, self._codebook_dict)
        return ids_dict, codebook_dict

    def ply_dtype(self, max_sh_degree: int):
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('rot_re', self.force_code_dtype or compute_uint_dtype(self.num_clusters_rotation_re)),
            ('rot_im', self.force_code_dtype or compute_uint_dtype(self.num_clusters_rotation_im)),
            ('opacity', self.force_code_dtype or compute_uint_dtype(self.num_clusters_opacity)),
            ('scale', self.force_code_dtype or compute_uint_dtype(self.num_clusters_scaling)),
            ('f_dc', self.force_code_dtype or compute_uint_dtype(self.num_clusters_features_dc)),
        ]
        for sh_degree in range(max_sh_degree):
            force_code_dtype = self.force_code_dtype or compute_uint_dtype(self.num_clusters_features_rest[sh_degree])
            dtype_full.extend([
                (f'f_rest_{sh_degree}_0', force_code_dtype),
                (f'f_rest_{sh_degree}_1', force_code_dtype),
                (f'f_rest_{sh_degree}_2', force_code_dtype),
            ])
        return dtype_full

    def ply_data(self, model: GaussianModel, ids_dict: Dict[str, torch.Tensor]):
        data_full = [
            *np.array_split(model._xyz.detach().cpu().numpy(), 3, axis=1),
            *np.array_split(torch.zeros_like(model._xyz).detach().cpu().numpy(), 3, axis=1),
            ids_dict["rotation_re"].unsqueeze(-1).cpu().numpy(),
            ids_dict["rotation_im"].unsqueeze(-1).cpu().numpy(),
            ids_dict["opacity"].unsqueeze(-1).cpu().numpy(),
            ids_dict["scaling"].unsqueeze(-1).cpu().numpy(),
            ids_dict["features_dc"].cpu().numpy(),
        ]
        for sh_degree in range(model.max_sh_degree):
            features_rest = ids_dict[f'features_rest_{sh_degree}'].cpu().numpy()
            data_full.extend(np.array_split(features_rest, 3, axis=1))
        return data_full

    def save_quantized(self, model: GaussianModel, ply_path: str):
        ids_dict, codebook_dict = self.quantize(model, update_codebook=False)
        dtype_full = self.ply_dtype(model.max_sh_degree)
        data_full = self.ply_data(model, ids_dict)

        elements = np.rec.fromarrays([data.squeeze(-1) for data in data_full], dtype=dtype_full)
        el = PlyElement.describe(elements, 'vertex')

        cb = [
            PlyElement.describe(array2record(codebook_dict["rotation_re"], "rot_re", 1, self.force_codebook_dtype), 'codebook_rot_re'),
            PlyElement.describe(array2record(codebook_dict["rotation_im"], "rot_im", 3, self.force_codebook_dtype), 'codebook_rot_im'),
            PlyElement.describe(array2record(codebook_dict["opacity"], "opacity", 1, self.force_codebook_dtype), 'codebook_opacity'),
            PlyElement.describe(array2record(codebook_dict["scaling"], "scaling", 3, self.force_codebook_dtype), 'codebook_scaling'),
            PlyElement.describe(array2record(codebook_dict["features_dc"], "f_dc", 3, self.force_codebook_dtype), 'codebook_f_dc'),
        ]
        for sh_degree in range(model.max_sh_degree):
            features_rest = codebook_dict[f'features_rest_{sh_degree}']
            n_channels = (sh_degree + 2) ** 2 - (sh_degree + 1) ** 2
            cb.append(PlyElement.describe(array2record(features_rest, f'f_rest_{sh_degree}', n_channels, self.force_codebook_dtype), f'codebook_f_rest_{sh_degree}'))

        PlyData([el, *cb]).write(ply_path)

    def parse_ids(self, plydata: PlyData, max_sh_degree: int, device: torch.device) -> Dict[str, torch.Tensor]:
        ids_dict = {}
        elements = plydata['vertex']
        kwargs = dict(dtype=torch.long, device=device)
        ids_dict["rotation_re"] = torch.tensor(elements["rot_re"].copy(), **kwargs)
        ids_dict["rotation_im"] = torch.tensor(elements["rot_im"].copy(), **kwargs)
        ids_dict["opacity"] = torch.tensor(elements["opacity"].copy(), **kwargs)
        ids_dict["scaling"] = torch.tensor(elements["scale"].copy(), **kwargs)
        ids_dict["features_dc"] = torch.tensor(elements["f_dc"].copy(), **kwargs).unsqueeze(-1)
        for sh_degree in range(max_sh_degree):
            ids_dict[f'features_rest_{sh_degree}'] = torch.tensor(np.stack([elements[f'f_rest_{sh_degree}_{ch}'] for ch in range(3)], axis=1), **kwargs)
        return ids_dict

    def parse_codebook(self, plydata: PlyData, max_sh_degree: int, device: torch.device) -> Dict[str, torch.Tensor]:
        codebook_dict = {}
        kwargs = dict(dtype=torch.float32, device=device)
        codebook_dict["rotation_re"] = torch.tensor(plydata["codebook_rot_re"]["rot_re"], **kwargs).unsqueeze(-1)
        codebook_dict["rotation_im"] = torch.tensor(np.stack([plydata["codebook_rot_im"][f'rot_im_{ch}'] for ch in range(3)], axis=1), **kwargs)
        codebook_dict["opacity"] = torch.tensor(plydata["codebook_opacity"]["opacity"], **kwargs).unsqueeze(-1)
        codebook_dict["scaling"] = torch.tensor(np.stack([plydata["codebook_scaling"][f'scaling_{ch}'] for ch in range(3)], axis=1), **kwargs)
        codebook_dict["features_dc"] = torch.tensor(np.stack([plydata["codebook_f_dc"][f'f_dc_{ch}'] for ch in range(3)], axis=1), **kwargs)
        for sh_degree in range(max_sh_degree):
            n_channels = (sh_degree + 2) ** 2 - (sh_degree + 1) ** 2
            codebook_dict[f'features_rest_{sh_degree}'] = torch.tensor(np.stack([plydata[f"codebook_f_rest_{sh_degree}"][f'f_rest_{sh_degree}_{ch}'] for ch in range(n_channels)], axis=1), **kwargs)
        return codebook_dict

    def parse_xyz(self, plydata: PlyData, device: torch.device) -> torch.Tensor:
        elements = plydata['vertex']
        kwargs = dict(dtype=torch.float32, device=device)
        xyz = torch.stack([
            torch.tensor(elements["x"].copy(), **kwargs),
            torch.tensor(elements["y"].copy(), **kwargs),
            torch.tensor(elements["z"].copy(), **kwargs),
        ], dim=1)
        return xyz

    def load_quantized(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        plydata = PlyData.read(ply_path)
        ids_dict = self.parse_ids(plydata, model.max_sh_degree, model._xyz.device)
        codebook_dict = self.parse_codebook(plydata, model.max_sh_degree, model._xyz.device)
        xyz = self.parse_xyz(plydata, model._xyz.device)
        self._codebook_dict = codebook_dict
        return self.dequantize(model, ids_dict, codebook_dict, xyz=xyz, replace=True)
