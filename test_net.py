"""
 A simple Pytorch implementation of GridNet, presented in Ref. 1. This
   implementation includes the modified version proposed in Ref. 2 (recommended
   for image-to-image translation).
 References:
   Ref. 1: Residual Conv-Deconv Grid Network for Semantic Segmentation,
     In BMVC, 2017.
   Ref. 2: Context-aware Synthesis for Video Frame Interpolation, In CVPR 2018.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import torch

from src import gridnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = gridnet.network(rows=3, columns=6, inchnls=3, outchnls=1,
                          norm=True, device=device)

data = torch.rand(1, 3, 128, 128).to(device=device)

output = network(data)

