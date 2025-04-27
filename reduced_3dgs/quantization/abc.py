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
    def dequantize(self, model: GaussianModel, ids_dict: Dict[str, torch.Tensor], codebook_dict: Dict[str, torch.Tensor], xyz: torch.Tensor = None, replace=False) -> GaussianModel:
        pass

    @abc.abstractmethod
    def save_quantized(self, model: GaussianModel, ply_path: str, codebook_path: str):
        pass

    @abc.abstractmethod
    def load_quantized(self, model: GaussianModel, ply_path: str, codebook_path: str) -> GaussianModel:
        pass


class QuantizeTrainerWrapper(TrainerWrapper, metaclass=abc.ABCMeta):
    def __init__(
            self, base_trainer: AbstractTrainer,
            quantizer: AbstractQuantizer,
            quantize_from_iter=5000,
            quantize_until_iter=30000,
            quantize_interval=1000,
    ):
        super().__init__(base_trainer)
        self.quantizer = quantizer
        self.quantize_from_iter = quantize_from_iter
        self.quantize_until_iter = quantize_until_iter
        self.quantize_interval = quantize_interval

    @property
    def model(self) -> GaussianModel:
        if self.quantize_from_iter <= self.curr_step <= self.quantize_until_iter and self.curr_step % self.quantize_interval == 0:
            with torch.no_grad():
                ids_dict, codebook_dict = self.quantizer.quantize(self.base_trainer.model, update_codebook=True)
                return self.quantizer.dequantize(self.base_trainer.model, ids_dict, codebook_dict)
        return self.base_trainer.model
