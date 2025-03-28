import abc
from typing import Dict, Tuple

import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper


class AbstractQuantizer(abc.ABC):

    @abc.abstractmethod
    def quantize(self, model: GaussianModel, update_codebook=True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pass

    @abc.abstractmethod
    def dequantize(self, model: GaussianModel, codebook_dict: Dict[str, torch.Tensor], ids_dict: Dict[str, torch.Tensor]) -> GaussianModel:
        pass

    @abc.abstractmethod
    def save_quantized(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass

    @abc.abstractmethod
    def load_quantized(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass


class QuantizeTrainerWrapper(TrainerWrapper, metaclass=abc.ABCMeta):
    def __init__(
            self, base_trainer: AbstractTrainer,
            quantizer: AbstractQuantizer,
            quantizate_from_iter=5000,
            quantizate_until_iter=30000,
            quantizate_interval=1000,
    ):
        super().__init__(base_trainer)
        self.quantizer = quantizer
        self.quantizate_from_iter = quantizate_from_iter
        self.quantizate_until_iter = quantizate_until_iter
        self.quantizate_interval = quantizate_interval

    @property
    def model(self) -> GaussianModel:
        if self.quantizate_from_iter <= self.curr_step < self.quantizate_until_iter and self.curr_step % self.quantizate_interval == 0:
            with torch.no_grad():
                ids_dict, codebook_dict = self.quantizer.quantize(self.base_trainer.model, update_codebook=True)
                return self.quantizer.dequantize(self.base_trainer.model, ids_dict, codebook_dict)
        return self.base_trainer.model
