import os
from typing import Dict
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from gaussian_splatting import GaussianModel
from plyfile import PlyData, PlyElement
import numpy as np
from .abc import AbstractVectorQuantizer


class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers

    def evaluate(self):
        return self.centers[self.ids.flatten().long()]


def generate_codebook(values: torch.Tensor, num_clusters=256, tol=0.0001, max_iter=500):
    kmeans = KMeans(
        n_clusters=num_clusters, tol=tol, max_iter=max_iter,
        init='random', random_state=0, n_init="auto", verbose=1,
        batch_size=256 * os.cpu_count()
    )
    ids = torch.tensor(kmeans.fit_predict(values.cpu().numpy()), device=values.device)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=values.dtype, device=values.device)
    return centers, ids


def produce_clusters(self: GaussianModel, num_clusters: int):
    codebook_dict: Dict[str, torch.Tensor] = {}
    ids_dict: Dict[str, torch.Tensor] = {}

    codebook_dict["features_dc"], ids = generate_codebook(self._features_dc.detach().squeeze(1), num_clusters=num_clusters, tol=0.001)
    ids_dict["features_dc"] = ids.unsqueeze(1)
    features_rest_flatten = self._features_rest.detach().transpose(1, 2).flatten(0, 1)
    for sh_degree in range(self.max_sh_degree):
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        codebook_dict[f"features_rest_{sh_degree}"], ids = generate_codebook(features_rest_flatten[:, sh_idx_start:sh_idx_end], num_clusters=num_clusters)
        ids_dict[f"features_rest_{sh_degree}"] = ids.reshape(-1, self._features_rest.shape[-1])

    codebook_dict["rotation_re"], ids_dict[f"rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1], num_clusters=num_clusters)
    codebook_dict["rotation_im"], ids_dict[f"rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:], num_clusters=num_clusters)

    codebook_dict["opacity"], ids_dict[f"opacity"] = generate_codebook(self._opacity.detach(), num_clusters=num_clusters)
    codebook_dict["scaling"], ids_dict[f"scaling"] = generate_codebook(self._scaling.detach(), num_clusters=num_clusters)
    return codebook_dict, ids_dict


def apply_clustering(self: GaussianModel, codebook_dict: Dict[str, torch.Tensor], ids_dict: Dict[str, torch.Tensor]):

    opacity = codebook_dict["opacity"][ids_dict[f"opacity"], ...]
    scaling = codebook_dict["scaling"][ids_dict[f"scaling"], ...]

    rotation = torch.cat((
        codebook_dict["rotation_re"][ids_dict[f"rotation_re"], ...],
        codebook_dict["rotation_im"][ids_dict[f"rotation_im"], ...],
    ), dim=1)

    features_rest = []
    for sh_degree in range(self.max_sh_degree):
        features_rest.append(codebook_dict[f"features_rest_{sh_degree}"][ids_dict[f"features_rest_{sh_degree}"], ...])
    features_rest = torch.cat(features_rest, dim=2).transpose(1, 2)

    features_dc = codebook_dict["features_dc"][ids_dict[f"features_dc"], ...]

    with torch.no_grad():
        self._opacity[...] = opacity
        self._scaling[...] = scaling
        self._rotation[...] = rotation
        self._features_dc[...] = features_dc
        self._features_rest[...] = features_rest
    return self


class VectorQuantizer(AbstractVectorQuantizer):
    def __init__(self, num_clusters=256):
        self.num_clusters = num_clusters

    def clustering(self, model: GaussianModel):
        codebook_dict, ids_dict = produce_clusters(model, self.num_clusters)
        return apply_clustering(model, codebook_dict, ids_dict)

    def save_clusters(self, model: GaussianModel, ply_path: str):
        codebook_dict = produce_clusters(model, self.num_clusters)
        opacity = codebook_dict["opacity"].ids
        scaling = codebook_dict["scaling"].ids
        rot = torch.cat((codebook_dict["rotation_re"].ids,
                        codebook_dict["rotation_im"].ids),
                        dim=1)
        features_dc = codebook_dict["features_dc"].ids
        features_rest = torch.cat([codebook_dict[f"features_rest_{i}"].ids
                                   for i in range(model.max_sh_degree)
                                   ], dim=1).squeeze()

        dtype_full = [(k, 'f4') for k in codebook_dict.keys()]
        codebooks = np.empty(self.num_clusters, dtype=dtype_full)

        centers_numpy_list = [v.centers.detach().cpu().numpy() for v in codebook_dict.values()]

        codebooks[:] = list(map(tuple, np.concatenate([ar for ar in centers_numpy_list], axis=1)))

        xyz = model._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = features_dc.cpu().numpy()
        f_rest = features_rest.flatten(start_dim=1).cpu().numpy()
        opacities = opacity.cpu().numpy()
        scale = scaling.cpu().numpy()
        rotation = rot.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]

        def construct_list_of_attributes(self: GaussianModel):
            l = []
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        dtype_full += [(attribute, 'u4') for attribute in construct_list_of_attributes(model)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        codebook_centers = PlyElement.describe(codebooks, 'codebook_centers')
        PlyData([el, codebook_centers]).write(ply_path)

    def load_clusters(self, model: GaussianModel, ply_path: str):
        pass
