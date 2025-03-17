from typing import List
import torch
from .quantizer import VectorQuantizer


class ExcludeZeroSHQuantizer(VectorQuantizer):

    def exclude_zero_feature_rest(self, codebook_dict, ids_dict):
        for sh_degree in range(self.model.max_sh_degree):
            codebook, ids = codebook_dict[f'features_rest_{sh_degree}'], ids_dict[f'features_rest_{sh_degree}']
            if codebook.abs().max() < 1e-6:  # invalid codebook
                codebook = torch.zeros_like(codebook[:1])
                ids[...] = 0
            codebook_dict[f'features_rest_{sh_degree}'], ids_dict[f'features_rest_{sh_degree}'] = codebook, ids
        return codebook_dict, ids_dict

    def produce_clusters(
            self,
            num_clusters_rotation_re: int,
            num_clusters_rotation_im: int,
            num_clusters_opacity: int,
            num_clusters_scaling: int,
            num_clusters_features_dc: int,
            num_clusters_features_rest: List[int],
            init_codebook_dict={}):
        for sh_degree in range(self.model.max_sh_degree):
            if f"features_rest_{sh_degree}" in init_codebook_dict and init_codebook_dict[f"features_rest_{sh_degree}"].abs().max() < 1e-6:
                init_codebook_dict[f"features_rest_{sh_degree}"] = None  # invalid codebook
        codebook_dict, ids_dict = super().produce_clusters(
            num_clusters_rotation_re=num_clusters_rotation_re,
            num_clusters_rotation_im=num_clusters_rotation_im,
            num_clusters_opacity=num_clusters_opacity,
            num_clusters_scaling=num_clusters_scaling,
            num_clusters_features_dc=num_clusters_features_dc,
            num_clusters_features_rest=num_clusters_features_rest,
            init_codebook_dict=init_codebook_dict
        )
        codebook_dict, ids_dict = self.exclude_zero_feature_rest(codebook_dict, ids_dict)
        return codebook_dict, ids_dict
