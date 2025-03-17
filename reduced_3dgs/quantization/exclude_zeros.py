import torch
from .quantizer import VectorQuantizer


class ExcludeZeroQuantizer(VectorQuantizer):

    @staticmethod
    def generate_codebook(values: torch.Tensor, num_clusters=256, treat_as_zero=1e-8, tol=1e-6, max_iter=500, init_codebook=None):
        zeros_mask = (values.abs() < treat_as_zero).all(-1)
        if zeros_mask.all():
            return torch.zeros(1, values.shape[1], dtype=values.dtype, device=values.device), torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
        if init_codebook is not None:
            if init_codebook.abs().max() < treat_as_zero:
                init_codebook = None
            elif init_codebook.shape[0] > num_clusters - 1 and init_codebook[0:-(num_clusters - 1), ...].abs().max() < treat_as_zero:
                init_codebook = init_codebook[-(num_clusters - 1):, ...]
        nonzero_values = values[~zeros_mask]
        nonzero_centers, nonzero_ids = super(ExcludeZeroQuantizer, ExcludeZeroQuantizer).generate_codebook(nonzero_values, num_clusters - 1, tol, max_iter, init_codebook)
        ids = torch.zeros(values.shape[0], dtype=nonzero_ids.dtype, device=nonzero_ids.device)
        ids[~zeros_mask] = nonzero_ids + 1
        centers = torch.cat((torch.zeros(1, values.shape[1], dtype=values.dtype, device=values.device), nonzero_centers), dim=0)
        return centers, ids
