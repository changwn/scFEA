# -*- coding: utf-8 -*-
"""

@author: wnchang
"""

# import sys
import torch
import torch.nn as nn


class FLUX(nn.Module):
    def __init__(self, n_modules, f_in=50, f_out=1):
        super(FLUX, self).__init__()
        # gene to flux
        self.inSize = f_in

        self.m_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.inSize, 8, bias=False),
                    nn.Tanhshrink(),
                    nn.Linear(8, f_out),
                    nn.Tanhshrink(),
                )
                for i in range(n_modules)
            ]
        )

    def forward(self, x, n_modules, n_genes, n_comps, cmMat):

        for i in range(n_modules):
            x_block = x[
                :,
                i * n_genes : (i + 1) * n_genes,
            ]
            subnet = self.m_encoder[i]
            if i == 0:
                m = subnet(x_block)
            else:
                m = torch.cat((m, subnet(x_block)), 1)

        c = m @ cmMat.T

        return m, c
