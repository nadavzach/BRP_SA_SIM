import torch
import torch.nn as nn
import copy
import cpy_smt_sa

from smt_sa.parallel_smt_sa import ParallelSMTSA


# Round implementation with straight-through estimator in backprop
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SystolicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', bypass=False):

        super(SystolicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self._x_max_val = 0
        self._x_min_val = 0
        self._x = None
        self._hw_sim = False
        self._gather_stats = False
        self._quantize = False
        self._bypass = bypass

    def set_hw_sim(self, val=True):
        self._hw_sim = val

    def gather_stats(self, val=True):
        self._gather_stats = val

    def quantize(self, val=True):
        self._quantize = val

    def _uniform_symmetric_quantization(self, x, x_min, x_max, bits):
        N = 2**bits
        delta = max(torch.abs(x_min), torch.abs(x_max)) * 2 / N
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, -N/2, N/2)
        return x_q * delta

    def forward(self, x):

        if self._bypass:
            return self.conv(x)

        if self._gather_stats:
            self._x_max_val = max(x.max().cpu(), self._x_max_val)
            self._x_min_val = min(x.min().cpu(), self._x_min_val)

        w_bak = None
        if self._quantize:
            # Activations quantization
            ##self._x_max_val = max(x.max(), self._x_max_val)
            ##self._x_min_val = min(x.min(), self._x_min_val)
            ##x = self._uniform_symmetric_quantization(x, self._x_min_val, self._x_max_val, 8)

            # Weights quantization
            w_max_val = self.conv.weight.data.max()
            w_min_val = self.conv.weight.data.min()
            w = self._uniform_symmetric_quantization(self.conv.weight.data, w_min_val, w_max_val, 8)

            w_bak = copy.deepcopy(self.conv.weight.data)
            self.conv.weight.data = w

            # TODO: quantize bias
            # Disable bias for now
            #b_bak = copy.deepcopy(self.conv.bias.data)
            #self.conv.bias.data = torch.zeros_like(self.conv.bias.data)

        if not self._hw_sim:
            out = self.conv(x)

        else:
            # Inspired from the example here:
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/fold.html#Unfold
            x_unf = nn.functional.unfold(x,
                                         kernel_size=(self.conv.kernel_size[0], self.conv.kernel_size[1]),
                                         padding=(self.conv.padding[0], self.conv.padding[1]),
                                         stride=(self.conv.stride[0], self.conv.stride[1])).transpose(1, 2)
            w_unf = self.conv.weight.view(self.conv.weight.size(0), -1).t()

            ofmap_height =\
                int((x.size(2) + 2 * self.conv.padding[0] - self.conv.kernel_size[0] + self.conv.stride[0])
                    / self.conv.stride[0])
            ofmap_width =\
                int((x.size(3) + 2 * self.conv.padding[1] - self.conv.kernel_size[1] + self.conv.stride[1])
                    / self.conv.stride[1])

            ##out_unf_ref = x_unf.matmul(w_unf).transpose(1, 2)

            # TEST
            threads = 2

            b_vec = w_unf.detach()
            a_vec = x_unf.reshape(x_unf.size(0) * x_unf.size(1), x_unf.size(2)).detach()
            a_vec = a_vec[:, :, None]
            a_vec = a_vec.expand(-1, -1, b_vec.size(1))

            psum = torch.mul(a_vec, b_vec)
            psum = psum != 0
            psum = psum.reshape(psum.size(0), threads, int(psum.size(1)/threads), psum.size(2))
            psum = psum.sum(dim=1)
            psum = psum.reshape(x_unf.size(0), x_unf.size(1), psum.size(1), psum.size(2))
            #psum = psum.cpu()


            #offset = torch.tensor((0, psum_bool_th_sum.size(1))).cuda()

            # Assuming no zero-valued weights
            quantized_indices = (psum[:, :, :, 0] == 2).nonzero()
            x_unf[quantized_indices[:, 0], quantized_indices[:, 1],
                  quantized_indices[:, 2]] =\
                torch.clamp(x_unf[quantized_indices[:, 0], quantized_indices[:, 1], quantized_indices[:, 2]], -1000, 15)
                #FloorSTE.apply(x_unf[quantized_indices[:, 0], quantized_indices[:, 1],
                #                     quantized_indices[:, 2]] / 16) * 16
            x_unf[quantized_indices[:, 0], quantized_indices[:, 1],
                  quantized_indices[:, 2] + int(x_unf.size(2) / threads)] =\
                torch.clamp(x_unf[quantized_indices[:, 0], quantized_indices[:, 1], quantized_indices[:, 2] + int(x_unf.size(2) / threads)], -1000, 15)
                #FloorSTE.apply(x_unf[quantized_indices[:, 0], quantized_indices[:, 1],
                #                     quantized_indices[:, 2] + int(x_unf.size(2) / threads)] / 16) * 16

            out_unf = x_unf.matmul(w_unf).transpose(1, 2)

            #x_unf = x_unf.cpu().detach().numpy()
            #w_unf = w_unf.cpu().detach().numpy()

            # Systolic array simulation
            # TODO: arguments should arrive from command line
            #sa = ParallelSMTSA(16, 2, 4)
            #sa.set_inputs(x_unf, w_unf)
            #out_unf = sa.go()

            #out_unf = torch.from_numpy(out_unf).float().cuda()
            #out_unf = out_unf.transpose(1, 2)

            # Check simulation output
            # TODO: should make the acceptable error in a define
            #assert(torch.max(torch.abs(out_unf - out_unf_ref)).cpu().item() < 1e-3)
            #print("Computation error ratio: {}".format(out_unf))

            out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
            # Adding the additional bias
            #out = out + self.conv.bias[None, :, None, None]

        # Recover previous not quantized weights
        if self._quantize:
            #self.conv.weight.data = w_bak
            #self.conv.bias.data = b_bak

            #out = out * x_delta * w_delta
            out = out + self.conv.bias[None, :, None, None]

        return out
