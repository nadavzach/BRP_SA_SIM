import torch
import torch.nn as nn
from QuantConv2d import UnfoldConv2d


class SimModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfold_list = []   # All UnfoldConv2d layers
        self.prune_list = []    # A subset of unfold_list to prune
        self.quant_list = []    # A subset of unfold_list to quantize (usually it's the same as unfold_list)

    def forward(self, x):
        raise NotImplementedError

    def update_unfold_list(self):
        self.apply(self._apply_unfold_list)

    def _apply_unfold_list(self, m):
        if type(m) == UnfoldConv2d:
            self.unfold_list.append(m)

    def bypass_all(self):
        for l in self.unfold_list:
            l._unfold = False
            l._prune = False
            l._quantize = False
            l._hw_sim = False
            l._reorder = None

    def prune_all(self):
        for l in self.prune_list:
            l._prune = True

    def quantize_all(self):
        for l in self.quant_list:
            l._quantize = True

    def disable_fc_grad(self):
        for name, param in self.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    def set_quantization_bits(self, x_bits, w_bits):
        for l in self.quant_list:
            l._x_bits = x_bits
            l._w_bits = w_bits

    def set_trunc_round(self):
        for l in self.unfold_list:
            l._round = True

    def set_sparsity(self, v):
        for l in self.unfold_list:
            l._exploit_sparsity = v

    def set_trunc_floor(self):
        for l in self.unfold_list:
            l._round = False

    def set_hw_type(self, type):
        #assert(type == 'wA' or type == 'A' or type == 'A0' or type == 'aW' or type == 'W' or type == 'W0')
        for l in self.unfold_list:
            l._hw_type = type

    def reset_stats(self):
        for l in self.unfold_list:
            l.reset_stats()
