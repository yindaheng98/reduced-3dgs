import torch
import math
from gaussian_splatting import GaussianModel, Camera

from gaussian_splatting.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.simple_knn._C import distCUDA2


class VariableSHBandsGaussianModel(GaussianModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._degrees = torch.empty(0)

    def to(self, device):
        self._degrees = self._degrees.to(device)
        return super().to(device)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        # TODO: compute SH according to self._degrees
        return torch.cat((features_dc, features_rest), dim=1)
