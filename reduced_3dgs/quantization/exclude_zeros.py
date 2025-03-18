import torch
from gaussian_splatting import GaussianModel
from .quantizer import VectorQuantizer


class ExcludeZeroSHQuantizer(VectorQuantizer):
    def __init__(self, model: GaussianModel, *args, treat_as_zero=1e-8, **kwargs):
        super(ExcludeZeroSHQuantizer, self).__init__(model, *args, **kwargs)
        self.treat_as_zero = treat_as_zero

    def generate_codebook_exclude_zero(self, values: torch.Tensor, num_clusters=256, init_codebook=None):
        zeros_mask = (values.abs() < self.treat_as_zero).all(-1)
        if zeros_mask.all():
            return torch.zeros(1, values.shape[1], dtype=values.dtype, device=values.device), torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
        if init_codebook is not None:
            if init_codebook.abs().max() < self.treat_as_zero:
                init_codebook = None
            elif init_codebook.shape[0] > num_clusters - 1:
                init_codebook = init_codebook[-(num_clusters - 1):, ...]
        nonzero_values = values[~zeros_mask]
        nonzero_centers, nonzero_ids = super().generate_codebook(nonzero_values, num_clusters - 1, init_codebook)
        ids = torch.zeros(values.shape[0], dtype=nonzero_ids.dtype, device=nonzero_ids.device)
        ids[~zeros_mask] = nonzero_ids + 1
        centers = torch.cat((torch.zeros(1, values.shape[1], dtype=values.dtype, device=values.device), nonzero_centers), dim=0)
        return centers, ids

    def produce_clusters_degree_features_rest(self, sh_degree, *args, **kwargs):
        features_rest_flatten = self.model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        codebook, ids = self.generate_codebook_exclude_zero(features_rest, self.num_clusters_features_rest[sh_degree], *args, **kwargs)
        return codebook, ids.reshape(-1, self.model._features_rest.shape[-1])
