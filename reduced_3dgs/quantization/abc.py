import abc
from gaussian_splatting import GaussianModel


class AbstractQuantizer:

    @abc.abstractmethod
    def produce_clusters(self, model: GaussianModel):
        pass

    @abc.abstractmethod
    def apply_clustering(self, model: GaussianModel):
        pass

    @abc.abstractmethod
    def load_ply(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass

    @abc.abstractmethod
    def save_ply(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass
