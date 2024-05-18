import matplotlib.pyplot as plt
import numpy as np
from torchio.transforms.preprocessing.intensity import NormalizationTransform
import torch
import pdb

class HUNormalization(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            HU_min_max = None,
            masking_method = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ('masking_method',)
        self.HU_min_max = HU_min_max

    def apply_normalization(
            self,
            subject,
            image_name: str,
            mask,
            ) -> None:
        image = subject[image_name]
        standardized = self.znorm(
            image.data,
            mask,
            self.HU_min_max,
        )
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        image.set_data(standardized)

    @staticmethod
    def znorm(tensor, mask, HU_min_max = None) -> torch.Tensor:
        tensor = tensor.clone().float()
        
        HU_min, HU_max = min(HU_min_max), max(HU_min_max)
        
        HU_min = torch.FloatTensor([HU_min])[0]
        HU_max = torch.FloatTensor([HU_max])[0]
        
        tensor = torch.where(tensor>HU_max, HU_max, tensor)
        tensor = torch.where(tensor<HU_min, HU_min, tensor)
        tensor = (tensor-HU_min)/(HU_max-HU_min)
        return tensor
