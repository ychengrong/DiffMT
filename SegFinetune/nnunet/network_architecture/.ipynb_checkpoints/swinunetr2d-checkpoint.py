import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETR2D(nn.Module):
    def __init__(self,img_size=(192, 192),in_channels=1,out_channels=2):
        super().__init__()
        self.swinUNETR = SwinUNETR(
                                img_size=img_size,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                feature_size=12,
                                use_checkpoint=True, 
                                spatial_dims=2
                                )

    
    def forward(self, x):
        x_out = self.swinUNETR(x)
        return x_out