import torch
from gaussian_splatting import GaussianModel


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
        # compute SH according to self._degrees
        n_SH = (self._degrees + 1) ** 2 - 1
        indices = torch.arange(n_SH.max(), device=n_SH.device).expand(n_SH.shape[0], -1) < n_SH.unsqueeze(-1)
        features_rest = torch.zeros_like(self._features_rest)
        features_rest[indices, :] = self._features_rest[indices, :]
        return torch.cat((features_dc, features_rest), dim=1)

    def init_degrees(self):
        self._degrees = torch.zeros(self._xyz.shape[0], dtype=torch.int, device=self._xyz.device) + self.max_sh_degree

    def create_from_pcd(self, *args, **kwargs):
        super().create_from_pcd(*args, **kwargs)
        self.init_degrees()

    def load_ply(self, *args, **kwargs):
        super().load_ply(*args, **kwargs)
        self.init_degrees()
