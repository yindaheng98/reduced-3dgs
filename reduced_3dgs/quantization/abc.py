import abc

from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper


class AbstractVectorQuantizer(TrainerWrapper):

    @abc.abstractmethod
    def clustering(self, model: GaussianModel):
        pass

    @abc.abstractmethod
    def save_clusters(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass

    @abc.abstractmethod
    def load_clusters(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass


class VectorQuantizeWrapper(TrainerWrapper, metaclass=abc.ABCMeta):
    def __init__(
            self, base_trainer: AbstractTrainer,
            quantizer: AbstractVectorQuantizer,
            quantizate_from_iter=15000,
            quantizate_until_iter=30000,
            quantizate_interval=3000,
    ):
        super().__init__(base_trainer)
        self.quantizer = quantizer
        self.quantizate_from_iter = quantizate_from_iter
        self.quantizate_until_iter = quantizate_until_iter
        self.quantizate_interval = quantizate_interval

    @property
    def model(self) -> GaussianModel:
        model = self.base_trainer.model
        if self.quantizate_from_iter <= self.curr_step < self.quantizate_until_iter and self.curr_step % self.quantizate_interval == 0:
            self.quantizer.clustering(model)
        return model
