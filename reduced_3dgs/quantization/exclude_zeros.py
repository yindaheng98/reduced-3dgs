import torch
from gaussian_splatting import GaussianModel
from .quantizer import VectorQuantizer


class ExcludeZeroQuantizer(VectorQuantizer):
    def __init__(self, model: GaussianModel, *args, treat_as_zero=1e-8, extract_zero_thr=0.5, **kwargs):
        super(ExcludeZeroQuantizer, self).__init__(model, *args, **kwargs)
        self.treat_as_zero = treat_as_zero
        self.extract_zero_thr = extract_zero_thr

    def generate_codebook(self, values: torch.Tensor, num_clusters=256, init_codebook=None):
        zeros_mask = (values.abs() < self.treat_as_zero).all(-1)
        if zeros_mask.all():
            return torch.zeros(1, values.shape[1], dtype=values.dtype, device=values.device), torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
        if zeros_mask.sum() <= self.extract_zero_thr * values.shape[0]:
            return super().generate_codebook(values, num_clusters, init_codebook)
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
