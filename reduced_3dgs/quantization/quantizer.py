from typing import Dict
import torch
import numpy as np
from gaussian_splatting import GaussianModel
from plyfile import PlyData, PlyElement
from reduced_3dgs.diff_gaussian_rasterization._C import kmeans_cuda
import numpy as np
from .abc import AbstractVectorQuantizer


class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers

    def evaluate(self):
        return self.centers[self.ids.flatten().long()]


def generate_codebook(values, inverse_activation_fn=lambda x: x, num_clusters=256, tol=0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1, 1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1, 1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))


def produce_clusters(self: GaussianModel, num_clusters: int):
    codebook_dict: Dict[str, Codebook] = {}

    codebook_dict["features_dc"] = generate_codebook(self._features_dc.detach()[:, 0],
                                                     num_clusters=num_clusters, tol=0.001)
    for sh_degree in range(self.max_sh_degree):
        sh_idx_start = (sh_degree + 1) ** 2 - 1
        sh_idx_end = (sh_degree + 2) ** 2 - 1
        codebook_dict[f"features_rest_{sh_degree}"] = generate_codebook(
            self._features_rest.detach()[:, sh_idx_start:sh_idx_end], num_clusters=num_clusters)

    codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                 self.inverse_opacity_activation, num_clusters=num_clusters)
    codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                 self.scaling_inverse_activation, num_clusters=num_clusters)
    codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                     num_clusters=num_clusters)
    codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                     num_clusters=num_clusters)
    return codebook_dict


def apply_clustering(self: GaussianModel, codebook_dict: Dict[str, Codebook]):
    max_coeffs_num = (self.max_sh_degree + 1)**2 - 1

    opacity = codebook_dict["opacity"].evaluate().requires_grad_(True)
    scaling = codebook_dict["scaling"].evaluate().view(-1, 3).requires_grad_(True)
    rotation = torch.cat((codebook_dict["rotation_re"].evaluate(),
                          codebook_dict["rotation_im"].evaluate().view(-1, 3)),
                         dim=1).squeeze().requires_grad_(True)
    features_dc = codebook_dict["features_dc"].evaluate().view(-1, 1, 3).requires_grad_(True)
    features_rest = []
    for sh_degree in range(max_coeffs_num):
        features_rest.append(codebook_dict[f"features_rest_{sh_degree}"].evaluate().view(-1, 3))

    features_rest = torch.stack([*features_rest], dim=1).squeeze().requires_grad_(True)

    with torch.no_grad():
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        self._features_dc = features_dc
        self._features_rest = features_rest
    return self


class VectorQuantizer(AbstractVectorQuantizer):
    def __init__(self, num_clusters=256):
        self.num_clusters = num_clusters

    def clustering(self, model: GaussianModel):
        codebook_dict = produce_clusters(model, self.num_clusters)
        return apply_clustering(model, codebook_dict)

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
