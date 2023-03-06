import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cpy_smt_sa as m
import numpy as np



class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StochasticRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        rand_mask = torch.rand_like(input)
        return torch.where((((input % 16) / 16) > rand_mask) & ((torch.round(input / 16) * 16) != 256) & (input > 15),
                           RoundSTE.apply(input / 16) * 16,
                           input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class MSBRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.where(input > 240,
                          torch.floor(input / 16) * 16,
                          input)
        out = torch.where((out > 15) & (((out % 16) / 16) > 0.5),
                          CeilSTE.apply(out / 16) * 16,
                          FloorSTE.apply(out / 16) * 16)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input)

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


def uniform_symmetric_quantization(x, x_min, x_max, bits):
    N = 2**bits
    delta = max(abs(x_min), abs(x_max)) * 2 / N
    x_int = RoundSTE.apply(x / delta)
    x_q = torch.clamp(x_int, -N/2, N/2 - 1)
    return x_q, delta


def uniform_symmetric_quantization_per_channel(x, x_min, x_max, bits):
    N = 2**bits
    delta = torch.where(x_min.abs() > x_max.abs(), x_min.abs(), x_max.abs()) * 2 / N
    x_int = RoundSTE.apply(x / delta[:, None, None, None].expand_as(x))
    x_q = torch.clamp(x_int, -N/2, N/2 - 1)
    return x_q, delta


def uniform_quantization(x, x_max, bits):
    N = 2**bits
    delta = x_max / N
    x_int = RoundSTE.apply(x / delta)
    x_q = torch.clamp(x_int, 0, N - 1)
    return x_q, delta


class UnfoldConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', unfold=True, prune=True, quantize=True):
        super(UnfoldConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        # Only saves the mask (used Conv2d so torch.save will save it as well)
        # In retrospective, I should have used register_buffer here instead
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.conv_mask.weight.data = torch.ones_like(self.conv_mask.weight.data)
        self.conv_mask.bias.data = torch.zeros_like(self.conv_mask.bias.data)
        # Disabling mask gradients, just in case
        for param in self.conv_mask.parameters():
            param.requires_grad = False

        # Registering buffers to be saved when calling torch.save
        self.register_buffer('num_inputs_tracked', torch.zeros(1))
        self.register_buffer('running_max_mean', torch.zeros(1))
        self.register_buffer('running_min_mean', torch.zeros(1))
        self.register_buffer('sort_stats', torch.zeros(self.conv.weight.size(1) *
                                                       self.conv.weight.size(2) *
                                                       self.conv.weight.size(3)))

        self.stats = {}
        self.stats['collisions'] = {4: {'total': 0, 'x_q': 0, 'w_q': 0},
                                    2: {'total': 0, 'x_q': 0, 'w_q': 0},
                                    1: {'total': 0, '8b-8b': 0, '4b-8b': 0, '8b-4b': 0, '4b-4b': 0},
                                    0: {'total': 0}}

        self.stats['inputs'] = 0
        self.stats['mse_error'] = 0
        self.stats['mac_count'] = 0
        self.stats['x_hist'] = 0

        self.stats['x'] = {'8b': 0, '4b': 0, '2b': 0, '0b': 0}
        self.stats['w'] = {'8b': 0, '4b': 0, '2b': 0, '0b': 0}

        self._unfold = unfold
        self._prune = prune
        self._quantize = quantize
        self._reorder = 'OFF'
        self._hw_sim = False
        self._x_bits = 8
        self._w_bits = 8
        self._round = False
        self._hw_type = 'wA'
        self._hw_mac = '2x4bx8b'
        self._threads = 2
        self._exploit_sparsity = None
        self._t1_analysis = None

        self._my_hw_sim = False

    def _masked_conv2d(self, x):
        return torch.nn.functional.conv2d(x, self.conv.weight * self.conv_mask.weight.data,
                                          bias=self.conv.bias.data if self.conv.bias is not None else None,
                                          stride=self.conv.stride, padding=self.conv.padding,
                                          dilation=self.conv.dilation, groups=self.conv.groups)

    def _reset_stats(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self._reset_stats(v)
            else:
                d[k] = 0

    def reset_stats(self):
        self._reset_stats(self.stats)

    def forward(self, x):

        x = x.type(torch.uint8)
        self.stats['inputs'] += x.size(0)
        if self._my_hw_sim:
            print(" $$$$$$$$ HW_SIM CPY_SMT_SA !START! $$$$$$$$$")
            
            weight_temp = self.conv.weight * self.conv_mask.weight.data if self._prune is True else self.conv.weight
            x_unf = nn.functional.unfold(x.to(torch.float32),
                                            kernel_size=(self.conv.kernel_size[0], self.conv.kernel_size[1]),
                                            padding=(self.conv.padding[0], self.conv.padding[1]),
                                            stride=(self.conv.stride[0], self.conv.stride[1])).transpose(1, 2).to(torch.uint8)
            w_unf = weight_temp.view(self.conv.weight.size(0), -1).t()
            w_np = w_unf.cpu().detach().numpy()
            x_np = x_unf.cpu().detach().numpy()
            #print("x - \n")
            #print(x_np.astype(np.uint))
            #print("w - \n")
            #print(x_np.astype(np.uint))
            temp_threads = 4
            sa_hw_sim_out = m.run_uint8(10,4,2,5000,x_np,w_np,True,False);
            out_unf = torch.from_numpy(sa_hw_sim_out[0]).transpose(1, 2).to(torch.float32)
            ofmap_height = \
                    int((x.size(2) + 2 * self.conv.padding[0] - self.conv.kernel_size[0] + self.conv.stride[0])
                        / self.conv.stride[0])
            ofmap_width = \
                    int((x.size(3) + 2 * self.conv.padding[1] - self.conv.kernel_size[1] + self.conv.stride[1])
                        / self.conv.stride[1])
            out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))

            stats_zero_ops              = sa_hw_sim_out[1]
            stats_1thread_mult_ops      = sa_hw_sim_out[2]
            stats_multi_thread_mult_ops = sa_hw_sim_out[3]
            stats_buffer_fullness_acc   = sa_hw_sim_out[4]
            stats_buffer_max_fullness   = sa_hw_sim_out[5]
            stats_alu_not_utilized      = sa_hw_sim_out[6]
            stats_total_cycles          = sa_hw_sim_out[7]
            #stats_speed_up = base_line_test_output[7] / stats_total_cycles
            stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;
            #mse_from_base_line = np.mean((result-base_line_test_output[0])**2)
            stats_alu_total = stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops + stats_alu_not_utilized

            print("finished layer, stats: \n")
            print("total cycles                     :  " +str(stats_total_cycles))
            #print("speed up from base line          :  " +str(stats_speed_up))
            print("stats_zero_ops %                 :  " +str(100*stats_zero_ops/stats_ops_total             ))
            print("stats_1thread_mult_ops %         :  " +str(100*stats_1thread_mult_ops/stats_ops_total     ))
            print("stats_multi_thread_mult_ops %    :  " +str(100*temp_threads*stats_multi_thread_mult_ops/stats_ops_total ))
            print("stats_total_thread_mult_ops %    :  " +str(stats_ops_total ))
            print("stats_buffer_fullness_acc        :  " +str(stats_buffer_fullness_acc  ))
            print("stats_buffer_max_fullness        :  " +str(stats_buffer_max_fullness  ))
            #print("MSE from base line               :  " +str(mse_from_base_line  ))
            print("stats_alu_not_utilized %         :  " +str(100*stats_alu_not_utilized/stats_alu_total ))
            print(" $$$$$$$$ HW_SIM CPY_SMT_SA !END! $$$$$$$$$")
            
        else:
            # Prepare activations, weights, and bias
            if self._quantize:
                # Gather statistics during training
                if self.training:
                    num_inputs_tracked_tag = self.num_inputs_tracked.clone()
                    self.num_inputs_tracked += x.size(0)
                    max_sum = x.detach().max(dim=3).values.max(dim=2).values.max(dim=1).values.sum()
                    min_sum = x.detach().min(dim=3).values.min(dim=2).values.min(dim=1).values.sum()

                    self.running_max_mean = \
                        ((self.running_max_mean * num_inputs_tracked_tag) + max_sum) / self.num_inputs_tracked
                    self.running_min_mean = \
                        ((self.running_min_mean * num_inputs_tracked_tag) + min_sum) / self.num_inputs_tracked

                # These statistics are mandatory for quantization
                    assert (self.running_max_mean != 0 or self.running_min_mean != 0)

                # Activations quantization
                # Currently only supports unsigned uniform quantization
                #print("$$$$$$$$$$$$ x min is: " + str(torch.min(x)))
                #print(torch.min(x).type())
                if torch.min(x) == 0:
                    x_q, x_q_delta = uniform_quantization(x, self.running_max_mean, self._x_bits)

                    if not self.training:
                        self.stats['x_hist'] += torch.histc(x_q, bins=256, min=0, max=255)
                        self.stats['x']['8b'] += x_q.numel()
                        self.stats['x']['4b'] += (x_q < 16).sum().item()
                        self.stats['x']['2b'] += (x_q < 4).sum().item()
                        self.stats['x']['0b'] += (x_q == 0).sum().item()
                else:
                    # x_q, x_q_delta = uniform_symmetric_quantization(x, self.running_min_mean, self.running_max_mean, 8)
                    raise NotImplementedError

                # Weights quantization
                weight_pre_q = self.conv.weight * self.conv_mask.weight.data if self._prune is True else self.conv.weight
                weight_q, weight_q_delta =\
                    uniform_symmetric_quantization_per_channel(weight_pre_q,
                                                            self.conv.weight.data.min(dim=3)[0].min(dim=2)[0].min(dim=1)[0],
                                                            self.conv.weight.data.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0], self._w_bits)

                # Bias quantization
                if self.conv.bias is None:
                    bias_fp = None
                else:
                    bias_q, bias_q_delta = uniform_symmetric_quantization(self.conv.bias,
                                                                        torch.min(self.conv.bias.data),
                                                                        torch.max(self.conv.bias.data), self._w_bits)
                    bias_fp = bias_q * bias_q_delta

                # Gather statistics
                if not self.training:
                    if self.stats['w']['8b'] == 0:
                        self.stats['w']['8b'] = weight_q.numel()
                        self.stats['w']['4b'] = ((weight_q < 8) & (weight_q >= -8)).sum().item()
                        self.stats['w']['2b'] = ((weight_q < 2) & (weight_q >= -2)).sum().item()
                        self.stats['w']['0b'] = (weight_q == 0).sum().item()

            else:
                x_q, x_q_delta = x, 1
                weight_q, weight_q_delta =\
                    self.conv.weight * self.conv_mask.weight.data if self._prune is True else self.conv.weight,\
                    torch.tensor([1])#.cuda()
                bias_fp = self.conv.bias

            if not self._unfold:
                out = nn.functional.conv2d(x_q * x_q_delta,
                                        weight_q * weight_q_delta[:, None, None, None].expand_as(weight_q),
                                        bias=bias_fp,
                                        stride=self.conv.stride[0],
                                        padding=self.conv.padding[0])
            else:
                #print(" == else (not self.unfold line 234 QuantCon2d")
                # At the moment, unfold and quantization must go together
                assert(self._quantize)

                x_unf = nn.functional.unfold(x_q,
                                            kernel_size=(self.conv.kernel_size[0], self.conv.kernel_size[1]),
                                            padding=(self.conv.padding[0], self.conv.padding[1]),
                                            stride=(self.conv.stride[0], self.conv.stride[1])).transpose(1, 2)
                w_unf = weight_q.view(self.conv.weight.size(0), -1).t()

                ofmap_height = \
                    int((x.size(2) + 2 * self.conv.padding[0] - self.conv.kernel_size[0] + self.conv.stride[0])
                        / self.conv.stride[0])
                ofmap_width = \
                    int((x.size(3) + 2 * self.conv.padding[1] - self.conv.kernel_size[1] + self.conv.stride[1])
                        / self.conv.stride[1])

                if self.training:
                    self.sort_stats += (x_unf > (2**(self._x_bits/2) - 1)).sum(dim=(0, 1))
                #print(" == hw sim is: " + str(self._hw_sim))

                if not self._hw_sim:
                    if self._my_hw_sim: # cpy_smt_sa hw simulation
                        print(" $$$$$$$$ HW_SIM CPY_SMT_SA !START! $$$$$$$$$")
                        x_unf_np = x_unf.cpu().detach().numpy()
                        w_unf_np = w_unf.cpu().detach().numpy()

                        sa_hw_sim_out = m.run_uint8(10,4,2,1000,x_unf_np,w_unf_np,True,False);
                        out_unf = torch.from_numpy(sa_hw_sim_out[0]).transpose(1, 2).to(torch.float32)

                        out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
                        
                        if bias_fp is None:
                            bias_fp = 0
                        else:
                            bias_fp = bias_fp[None, :, None, None].expand_as(out)

                        out = out * x_q_delta * weight_q_delta[None, :, None, None].expand_as(out) + bias_fp

                        stats_zero_ops              = sa_hw_sim_out[1]
                        stats_1thread_mult_ops      = sa_hw_sim_out[2]
                        stats_multi_thread_mult_ops = sa_hw_sim_out[3]
                        stats_buffer_fullness_acc   = sa_hw_sim_out[4]
                        stats_buffer_max_fullness   = sa_hw_sim_out[5]
                        stats_alu_not_utilized      = sa_hw_sim_out[6]
                        stats_total_cycles          = sa_hw_sim_out[7]
                        #stats_speed_up = base_line_test_output[7] / stats_total_cycles
                        stats_ops_total = stats_zero_ops + stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops;
                        #mse_from_base_line = np.mean((result-base_line_test_output[0])**2)
                        stats_alu_total = stats_1thread_mult_ops + 2*stats_multi_thread_mult_ops + stats_alu_not_utilized

                        print("finished layer, stats: \n")
                        print("total cycles                     :  " +str(stats_total_cycles))
                        #print("speed up from base line          :  " +str(stats_speed_up))
                        print("stats_zero_ops %                 :  " +str(100*stats_zero_ops/stats_ops_total             ))
                        print("stats_1thread_mult_ops %         :  " +str(100*stats_1thread_mult_ops/stats_ops_total     ))
                        print("stats_multi_thread_mult_ops %    :  " +str(100*4*stats_multi_thread_mult_ops/stats_ops_total ))
                        print("stats_total_thread_mult_ops %    :  " +str(stats_ops_total ))
                        print("stats_buffer_fullness_acc        :  " +str(stats_buffer_fullness_acc  ))
                        print("stats_buffer_max_fullness        :  " +str(stats_buffer_max_fullness  ))
                        #print("MSE from base line               :  " +str(mse_from_base_line  ))
                        print("stats_alu_not_utilized %         :  " +str(100*stats_alu_not_utilized/stats_alu_total ))
                        print(" $$$$$$$$ HW_SIM CPY_SMT_SA !END! $$$$$$$$$")
                    else: # no hw simulation

                        print(" $$$$$$$$ saving a and b $$$$$$$$$")
                        out_unf = x_unf.matmul(w_unf).transpose(1, 2)
                        out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
                        a_w = 50
                        a_h = 50
                        a_c = 50
                        b_w = a_c
                        b_h = 50
                        mean = np.mean(x_unf)
                        variance = np.var(x_unf)
                        zero_percent = np.count_nonzero(x_unf == 0) / np.prod(x_unf.shape)
                        values = np.random.normal(loc=mean,scale= np.sqrt(variance),size= (a_w*a_h*a_c))
                        values = np.reshape(values,(a_w,a_h,a_c))

                        a_num_zeros = int(zero_percent * a_w * a_h*a_c )
                        a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
                        values.ravel()[a_zero_indices] = 0
                        new_zero_per = np.count_nonzero(values == 0) / np.prod(values.shape)
                        print("calc zero per = "+str(zero_percent)+", new zero per is "+str(new_zero_per)+" \n")

                        np.savez_compressed("{}/{}.npz".format('./src/cpy_smt/test/a_b_mats', 'a_mat'+str(self._get_name())), values)
                        mean = np.mean(w_unf)
                        variance = np.var(w_unf)
                        zero_percent = np.count_nonzero(w_unf == 0) / np.prod(w_unf.shape)
                        values = np.random.normal(loc=mean,scale= np.sqrt(variance),size= (b_w*b_h))
                        values = np.reshape(values,(b_w,b_h))

                        b_num_zeros = int(zero_percent * b_w * b_h )
                        b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
                        values.ravel()[a_zero_indices] = 0
                        new_zero_per = np.count_nonzero(values == 0) / np.prod(values.shape)
                        print("calc zero per = "+str(zero_percent)+", new zero per is "+str(new_zero_per)+" \n")

                        np.savez_compressed("{}/{}.npz".format('./src/cpy_smt/test/a_b_mats', 'b_mat'+str(self._get_name())), values)
 

                        if bias_fp is None:
                            bias_fp = 0
                        else:
                            bias_fp = bias_fp[None, :, None, None].expand_as(out)

                        out = out * x_q_delta * weight_q_delta[None, :, None, None].expand_as(out) + bias_fp
                # HW simulation
                else:
                    print(" == hw sim start == ")
                    threads = self._threads
                    assert (threads == 4 or threads == 2 or threads == 1)
                    trunc_func = RoundSTE if self._round is True else FloorSTE

                    # Data reorder
                    if self._reorder != 'OFF':
                        if self._reorder == 'IDEAL':
                            _, sort_idx = (x_unf > (2**(self._x_bits/2) - 1)).sum(dim=(0, 1)).sort()
                        elif self._reorder == 'STATS':
                            assert (self.sort_stats.sum().item() != 0)
                            _, sort_idx = self.sort_stats.sort()
                        elif self._reorder == 'WEIGHTS':
                            _, sort_idx = ((w_unf > 7) | (w_unf < -8)).sum(dim=1).sort()
                            #_, sort_idx = w_unf.abs().sum(dim=1).sort()
                        else:
                            raise NotImplementedError

                        if threads == 2:
                            sort_idx[int(sort_idx.size(0) / 2):] = sort_idx[int(sort_idx.size(0) / 2):].__reversed__()
                        elif threads == 4:
                            sort_idx[int(sort_idx.size(0) / 4):int(2 * sort_idx.size(0) / 4) - 1] =\
                                sort_idx[int(sort_idx.size(0) / 4):int(2 * sort_idx.size(0) / 4) - 1].__reversed__()
                            sort_idx[int(3 * sort_idx.size(0) / 4):] =\
                                sort_idx[int(3 * sort_idx.size(0) / 4):].__reversed__()

                        x_unf = x_unf[:, :, sort_idx]
                        w_unf = w_unf[sort_idx, :]

                    # TODO: enable
                    if (x_unf.size(2) % threads) != 0:
                        #raise AssertionError
                        x_unf = torch.cat((x_unf, torch.zeros(x_unf.size(0), x_unf.size(1), 1)), dim=2)
                        w_unf = torch.cat((w_unf, torch.zeros(1, w_unf.size(1))))

                    aaa = x_unf.reshape(x_unf.size(0), x_unf.size(1), threads, int(x_unf.size(2) / threads))[:, :, None, :, :] \
                        .expand([-1, -1, w_unf.size(1), -1, -1]) \
                        .contiguous()
                    bbb = w_unf.t().reshape(w_unf.size(1), threads, int(w_unf.size(0) / threads))[None, None, :, :, :] \
                        .expand(x_unf.size(0), x_unf.size(1), -1, -1, -1).contiguous()

                    non_zero_elems = ((aaa.detach() * bbb.detach()) != 0).sum(dim=3)

                    # We merge both 4 threads and 3 threads stats
                    if self._threads == 4:
                        self.stats['collisions'][4]['total'] += (non_zero_elems == 4).sum().item() +\
                                                                (non_zero_elems == 3).sum().item()
                    if self._threads >= 2:
                        self.stats['collisions'][2]['total'] += (non_zero_elems == 2).sum().item()
                    self.stats['collisions'][1]['total'] += (non_zero_elems == 1).sum().item()
                    self.stats['collisions'][0]['total'] += (non_zero_elems == 0).sum().item()

                    if self._hw_mac == '2x4bx8b':
                        if threads == 4:
                            # If sparsity is exploited then mark collisions
                            if self._exploit_sparsity:
                                coll = ((non_zero_elems == 4) | (non_zero_elems == 3))[:, :, :, None, :].expand(
                                    [-1, -1, -1, threads, -1])
                            # Otherwise, it is equivalent for having collision everywhere
                            else:
                                coll = torch.ones_like(aaa).bool()

                            # In the 4-bit case we do not differentiate between A0 and W0
                            qqq = coll if (self._hw_type == 'A0' or self._hw_type == 'W0') else (coll & (aaa > 15))
                            aaa = torch.where(qqq == True, trunc_func.apply(aaa / 16) * 16, aaa)
                            self.stats['collisions'][4]['x_q'] += qqq.sum().item()

                            qqq = coll if (self._hw_type == 'A0' or self._hw_type == 'W0') else\
                                (coll & ((bbb > 7) | (bbb < -8)))
                            bbb = torch.where(qqq == True, trunc_func.apply(bbb / 16) * 16, bbb)
                            self.stats['collisions'][4]['w_q'] += qqq.sum().item()

                        if (threads == 2) or (threads == 4 and self._exploit_sparsity):
                            if self._exploit_sparsity:
                                coll = (non_zero_elems == 2)[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
                            else:
                                coll = torch.ones_like(aaa).bool()

                            if self._hw_type == 'wA' or self._hw_type == 'aW':
                                qqq = coll & ((aaa > 15) & ((bbb > 7) | (bbb < -8)))
                            elif self._hw_type == 'A':
                                qqq = coll & (aaa > 15)
                            elif self._hw_type == 'W':
                                qqq = coll & ((bbb > 7) | (bbb < -8))
                            elif self._hw_type == 'A0' or self._hw_type == 'W0':
                                qqq = coll
                            else:
                                raise NotImplementedError

                            if self._hw_type == 'wA' or self._hw_type == 'A' or self._hw_type == 'A0':
                                aaa = torch.where(qqq == True, trunc_func.apply(aaa / 16) * 16, aaa)

                                try:
                                    self.stats['collisions'][2]['x_q'] += qqq.sum().item()
                                except:
                                    self.stats['collisions'][2]['x_q'] += qqq.cpu().sum().item()
                            elif self._hw_type == 'aW' or self._hw_type == 'W' or self._hw_type == 'W0':
                                bbb = torch.where(qqq == True, trunc_func.apply(bbb / 16) * 16, bbb)

                                try:
                                    self.stats['collisions'][2]['w_q'] += qqq.sum().item()
                                except:
                                    self.stats['collisions'][2]['w_q'] += qqq.cpu().sum().item()
                            else:
                                raise NotImplementedError

                        if self._t1_analysis:
                            assert (threads == 1)

                            coll = (non_zero_elems == 1)[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
                            qqq = coll & (aaa > 15) & ((bbb > 7) | (bbb < -8))
                            self.stats['collisions'][1]['8b-8b'] += qqq.sum().item()

                            qqq = coll & (aaa <= 15) & ((bbb > 7) | (bbb < -8))
                            self.stats['collisions'][1]['4b-8b'] += qqq.sum().item()

                            qqq = coll & (aaa > 15) & ((bbb <= 7) & (bbb >= -8))
                            self.stats['collisions'][1]['8b-4b'] += qqq.sum().item()

                            qqq = coll & (aaa <= 15) & ((bbb <= 7) & (bbb >= -8))
                            self.stats['collisions'][1]['4b-4b'] += qqq.sum().item()

                    else:
                        raise NotImplementedError

                    # Finalizing multiplications
                    data_tensor = aaa * bbb
                    data_tensor = data_tensor.sum(dim=(3, 4))
                    out_unf = data_tensor.transpose(1, 2)

                    out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
                    out = out * x_q_delta * weight_q_delta[None, :, None, None].expand_as(out) + (0 if bias_fp is None else bias_q[None, :, None, None] * bias_q_delta)

                    out_ref = nn.functional.conv2d(x_q * x_q_delta,
                                                weight_q * weight_q_delta[:, None, None, None].expand_as(weight_q),
                                                bias=None if bias_fp is None else bias_q * bias_q_delta,
                                                stride=self.conv.stride[0],
                                                padding=self.conv.padding[0]).detach()

                    self.stats['mse_error'] += torch.nn.functional.mse_loss(out, out_ref).item()

                    print(" == hw sim end == ")

        if not self.training:
            self.stats['mac_count'] += out.size(0) * out.size(1) * out.size(2) * out.size(3) * \
                                       self.conv.weight.size(1) * self.conv.weight.size(2) * self.conv.weight.size(3)

        #print("quantConv2d end")
        return out
