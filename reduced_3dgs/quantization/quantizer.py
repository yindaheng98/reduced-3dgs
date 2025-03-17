import os
from typing import Dict, List
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
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
            self, model: GaussianModel,
            num_clusters=256,
            num_clusters_rotation_re=None,
            num_clusters_rotation_im=None,
            num_clusters_opacity=None,
            num_clusters_scaling=None,
            num_clusters_features_dc=None,
            num_clusters_features_rest=[],
            force_code_dtype=None,
            force_codebook_dtype='f4',
    ):
        self._model = model
        self.num_clusters_rotation_re = num_clusters_rotation_re or num_clusters
        self.num_clusters_rotation_im = num_clusters_rotation_im or num_clusters
        self.num_clusters_opacity = num_clusters_opacity or num_clusters
        self.num_clusters_scaling = num_clusters_scaling or num_clusters
        self.num_clusters_features_dc = num_clusters_features_dc or num_clusters
        self.num_clusters_features_rest = [(num_clusters_features_rest[i] if len(num_clusters_features_rest) > i else num_clusters) for i in range(model.max_sh_degree)]
        self.force_code_dtype = force_code_dtype
        self.force_codebook_dtype = force_codebook_dtype
        self._codebook_dict = {}

    @property
    def model(self) -> GaussianModel:
        return self._model

    @staticmethod
    def generate_codebook(values: torch.Tensor, num_clusters=256, tol=1e-6, max_iter=500, init_codebook=None):
        kmeans = KMeans(
            n_clusters=num_clusters, tol=tol, max_iter=max_iter,
            init='random' if init_codebook is None else init_codebook.cpu().numpy(),
            random_state=0, n_init="auto", verbose=0,
            batch_size=256 * os.cpu_count()
        )
        ids = torch.tensor(kmeans.fit_predict(values.cpu().numpy()), device=values.device)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=values.dtype, device=values.device)
        return centers, ids

    def produce_clusters_features_dc(self, *args, **kwargs):
        codebook, ids = self.generate_codebook(self.model._features_dc.detach().squeeze(1), *args, **kwargs)
        return codebook, ids.unsqueeze(1)

    def produce_clusters_degree_features_rest(self, sh_degree, *args, **kwargs):
        features_rest_flatten = self.model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        codebook, ids = self.generate_codebook(features_rest, *args, **kwargs)
        return codebook, ids.reshape(-1, self.model._features_rest.shape[-1])

    def produce_clusters_rotation_re(self, *args, **kwargs):
        return self.generate_codebook(self.model.get_rotation.detach()[:, 0:1], *args, **kwargs)

    def produce_clusters_rotation_im(self, *args, **kwargs):
        return self.generate_codebook(self.model.get_rotation.detach()[:, 1:], *args, **kwargs)

    def produce_clusters_opacity(self, *args, **kwargs):
        return self.generate_codebook(self.model._opacity.detach(), *args, **kwargs)

    def produce_clusters_scaling(self, *args, **kwargs):
        return self.generate_codebook(self.model.get_scaling.detach(), *args, **kwargs)

    def produce_clusters(
            self,
            num_clusters_rotation_re: int,
            num_clusters_rotation_im: int,
            num_clusters_opacity: int,
            num_clusters_scaling: int,
            num_clusters_features_dc: int,
            num_clusters_features_rest: List[int],
            init_codebook_dict={}):

        codebook_dict: Dict[str, torch.Tensor] = {}
        ids_dict: Dict[str, torch.Tensor] = {}

        init_codebook_dict = {
            "features_dc": None,
            **{f"features_rest_{sh_degree}": None for sh_degree in range(self.model.max_sh_degree)},
            "rotation_re": None,
            "rotation_im": None,
            "opacity": None,
            "scaling": None,
            **init_codebook_dict
        }

        codebook_dict["features_dc"], ids_dict["features_dc"] = self.produce_clusters_features_dc(num_clusters_features_dc, init_codebook=init_codebook_dict["features_dc"])
        for sh_degree in range(self.model.max_sh_degree):
            codebook_dict[f"features_rest_{sh_degree}"], ids_dict[f"features_rest_{sh_degree}"] = self.produce_clusters_degree_features_rest(
                sh_degree, num_clusters_features_rest[sh_degree], init_codebook=init_codebook_dict[f"features_rest_{sh_degree}"]
            )
        codebook_dict["rotation_re"], ids_dict[f"rotation_re"] = self.produce_clusters_rotation_re(num_clusters_rotation_re, init_codebook=init_codebook_dict["rotation_re"])
        codebook_dict["rotation_im"], ids_dict[f"rotation_im"] = self.produce_clusters_rotation_im(num_clusters_rotation_im, init_codebook=init_codebook_dict["rotation_im"])
        codebook_dict["opacity"], ids_dict[f"opacity"] = self.produce_clusters_opacity(num_clusters_opacity, init_codebook=init_codebook_dict["opacity"])
        codebook_dict["scaling"], ids_dict[f"scaling"] = self.produce_clusters_scaling(num_clusters_scaling, init_codebook=init_codebook_dict["scaling"])
        return codebook_dict, ids_dict

    def apply_clustering(
            self,
            codebook_dict: Dict[str, torch.Tensor],
            ids_dict: Dict[str, torch.Tensor]):
        model = self.model
        opacity = codebook_dict["opacity"][ids_dict["opacity"], ...]
        scaling = model.scaling_inverse_activation(codebook_dict["scaling"][ids_dict["scaling"], ...])

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
            model._opacity[...] = opacity
            model._scaling[...] = scaling
            model._rotation[...] = rotation
            model._features_dc[...] = features_dc
            model._features_rest[...] = features_rest
        return model

    def quantize(self) -> GaussianModel:
        codebook_dict, ids_dict = self.produce_clusters(
            self.num_clusters_rotation_re,
            self.num_clusters_rotation_im,
            self.num_clusters_opacity,
            self.num_clusters_scaling,
            self.num_clusters_features_dc,
            self.num_clusters_features_rest,
            self._codebook_dict)
        self._codebook_dict = codebook_dict
        return self.apply_clustering(codebook_dict, ids_dict)

    def save_quantized(self, ply_path: str):
        model = self.model
        codebook_dict, ids_dict = self.produce_clusters(
            self.num_clusters_rotation_re,
            self.num_clusters_rotation_im,
            self.num_clusters_opacity,
            self.num_clusters_scaling,
            self.num_clusters_features_dc,
            self.num_clusters_features_rest,
            self._codebook_dict)
        self._codebook_dict = codebook_dict
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('rot_re', self.force_code_dtype or compute_uint_dtype(self.num_clusters_rotation_re)),
            ('rot_im', self.force_code_dtype or compute_uint_dtype(self.num_clusters_rotation_im)),
            ('opacity', self.force_code_dtype or compute_uint_dtype(self.num_clusters_opacity)),
            ('scale', self.force_code_dtype or compute_uint_dtype(self.num_clusters_scaling)),
            ('f_dc', self.force_code_dtype or compute_uint_dtype(self.num_clusters_features_dc)),
        ]
        for sh_degree in range(model.max_sh_degree):
            force_code_dtype = self.force_code_dtype or compute_uint_dtype(self.num_clusters_features_rest[sh_degree])
            dtype_full.extend([
                (f'f_rest_{sh_degree}_0', force_code_dtype),
                (f'f_rest_{sh_degree}_1', force_code_dtype),
                (f'f_rest_{sh_degree}_2', force_code_dtype),
            ])
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

        return self.apply_clustering(codebook_dict, ids_dict)

    def load_quantized(self, ply_path: str):
        model = self.model
        plydata = PlyData.read(ply_path)

        ids_dict = {}
        elements = plydata['vertex']
        kwargs = dict(dtype=torch.long, device=model._xyz.device)
        ids_dict["rotation_re"] = torch.tensor(elements["rot_re"].copy(), **kwargs)
        ids_dict["rotation_im"] = torch.tensor(elements["rot_im"].copy(), **kwargs)
        ids_dict["opacity"] = torch.tensor(elements["opacity"].copy(), **kwargs)
        ids_dict["scaling"] = torch.tensor(elements["scale"].copy(), **kwargs)
        ids_dict["features_dc"] = torch.tensor(elements["f_dc"].copy(), **kwargs).unsqueeze(-1)
        for sh_degree in range(model.max_sh_degree):
            ids_dict[f'features_rest_{sh_degree}'] = torch.tensor(np.stack([elements[f'f_rest_{sh_degree}_{ch}'] for ch in range(3)], axis=1), **kwargs)

        codebook_dict = {}
        kwargs = dict(dtype=torch.float32, device=model._xyz.device)
        codebook_dict["rotation_re"] = torch.tensor(plydata["codebook_rot_re"]["rot_re"], **kwargs).unsqueeze(-1)
        codebook_dict["rotation_im"] = torch.tensor(np.stack([plydata["codebook_rot_im"][f'rot_im_{ch}'] for ch in range(3)], axis=1), **kwargs)
        codebook_dict["opacity"] = torch.tensor(plydata["codebook_opacity"]["opacity"], **kwargs).unsqueeze(-1)
        codebook_dict["scaling"] = torch.tensor(np.stack([plydata["codebook_scaling"][f'scaling_{ch}'] for ch in range(3)], axis=1), **kwargs)
        codebook_dict["features_dc"] = torch.tensor(np.stack([plydata["codebook_f_dc"][f'f_dc_{ch}'] for ch in range(3)], axis=1), **kwargs)
        for sh_degree in range(model.max_sh_degree):
            n_channels = (sh_degree + 2) ** 2 - (sh_degree + 1) ** 2
            codebook_dict[f'features_rest_{sh_degree}'] = torch.tensor(np.stack([plydata[f"codebook_f_rest_{sh_degree}"][f'f_rest_{sh_degree}_{ch}'] for ch in range(n_channels)], axis=1), **kwargs)

        self._codebook_dict = codebook_dict
        return self.apply_clustering(codebook_dict, ids_dict)
