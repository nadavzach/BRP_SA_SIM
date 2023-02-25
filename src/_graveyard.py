def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
            print((grad == 0).sum().item() / grad.numel())
        except AttributeError:
            print("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")
    print("\n")


def smt_sa_debug(test=None):
    # Random results comparison
    if test == 'RAND':
        for i in range(10000):
            d1 = np.random.randint(4) + 1
            d2 = np.random.randint(32) + 1
            d3 = np.random.randint(4096) + 1
            d4 = np.random.randint(32) + 1
            a = np.random.randint(32, size=(d1, d2, d3)) - 16
            b = np.random.randint(32, size=(d3, d4)) - 16
            result = cpy_smt_sa.run_int64(8, 2, 4, a, b)
            print("({},{},{})x({},{})".format(d1, d2, d3, d3, d4))

            if np.sum(np.matmul(a, b) - result) != 0:
                print(a)
                print(b)
                print(result)
                print(np.matmul(a, b))
                print("Failed")
                exit()
            else:
                print("Passed {}".format(i))

        return

    # Performance analysis (FP32)
    if test == 'PERF':
        test_acc = 0
        macs = 10000000
        start = time.time()
        for i in range(macs):
            test_acc += i * i
        end = time.time()
        exec_time = end - start
        ideal_mops_s = macs/exec_time/1024/1024
        print('IDEAL: {} s, {} Mops/s'.format(exec_time, ideal_mops_s))

        a, b, c, d, e = 4, 32, 256, 256, 32
        dim, threads, max_depth = 16, 2, 4096
        macs = a * b * d * e

        aa = np.random.randn(a, b, c)
        bb = np.random.randn(d, e)
        result_ref = np.matmul(aa, bb)

        # CPY_SMT_SA
        start = time.time()
        result = cpy_smt_sa.run_fp32(dim, threads, max_depth, aa, bb)
        end = time.time()

        assert (np.max(np.abs(result_ref - result)) < 1e-3)
        exec_time = end - start
        t_cpy_smt_sa = exec_time
        print('CPY_SMT_SA: {} s, {} Mops/s'.format(exec_time, macs/exec_time/1024/1024))

        # PY_SMT_SA
        start = time.time()
        sa = SMTSA(dim, threads, max_depth)
        sa.set_inputs(aa, bb)
        sa.go()
        end = time.time()

        exec_time = end - start
        t_py_smt_sa = exec_time
        print('PY_SMT_SA: {} s, {} Mops/s'.format(exec_time, macs/exec_time/1024/1024))
        print('C++/Python speed: {}'.format(t_py_smt_sa / t_cpy_smt_sa))
        print('Python simulator overhead: {}'.format(ideal_mops_s / (macs/exec_time/1024/1024)))

        return



    # -------------
    # - GRAVEYARD -
    # -------------

    #cfg.LOG.start_new_log(name='{}-{}-train'.format(arch, dataset))
    #nn = NeuralNet(arch, dataset)
    #nn = NeuralNet(arch, dataset, model_chkp=model_chkp)
    #nn.next_train_epoch = 0
    #nn.test(test_gen)
    #nn.train(train_gen, test_gen, epochs=120, lr=0.001, lr_plan={30: 0.0001}, wd=0.0005)
    #nn.train(train_gen, test_gen, epochs=120, lr=0.01, lr_plan={30: 0.01, 60: 0.0001, 90: 0.00001}, wd=0.0005)
    #cfg.LOG.close_log()
    #return

    # Unstructured weights pruning
    cfg.LOG.start_new_log(name='{}-{}-w_prune'.format(arch, dataset))
    #nn = NeuralNet(arch, dataset, model_chkp='/home/gilsho/approx_sa/src/data/results/_alexnet_imagenet_epoch-3_wprune-35_quant-8_top1-56.25.pth')
    nn = NeuralNet(arch, dataset, model_chkp='/home/gilsho/approx_sa/src/data/results/_resnet18_imagenet_epoch-4_wquant-50_top1-69.19.pth')
    #nn = NeuralNet(arch, dataset)

    for name, param in nn.model.named_parameters():
        print(name, param.requires_grad)
        if 'fc' in name:
            param.requires_grad = False
            print('Set {} requires_grad to {}'.format(name, param.requires_grad))

    ###nn.model.conv2.register_backward_hook(hook_fn)

    nn.next_train_epoch = 0
    nn.best_top1_acc = 0
    #nn.test(test_gen)
    #nn.train(train_gen, test_gen, epochs=5, lr=0.000001, desc='wprune-35_quant-8') # AlexNet
    nn.train(train_gen, test_gen, epochs=10, lr=0.00001, wd=0.0001, desc='wprune-35_quant-8') # ResNet-18
    #nn.train(train_gen, test_gen, epochs=1, lr=0, desc='wprune-35_quant-8')
    return



    for quant in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        nn.best_top1_acc = 0
        for l in nn.model.prune_list:
            th = np.quantile(l.conv.weight.data.abs().detach().cpu().numpy(), quant/100)
            l.conv_mask.weight.data = torch.gt(l.conv.weight.data.abs(), th).float()

        #nn.train(train_gen, test_gen, epochs=5, lr=0.00001, wd=0.0005, desc='wquant-{}'.format(quant)) # AlexNet
        nn.train(train_gen, test_gen, epochs=5, lr=0.0001, wd=0.0001, desc='wquant-{}'.format(quant)) # ResNet-18
    cfg.LOG.close_log()
    return

    # Low precision tests
    cfg.LOG.start_new_log(name='{}-{}-lowp'.format(arch, dataset))
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)
    nn.next_train_epoch = 0
    #nn.train(train_gen, test_gen, epochs=120, lr=0.001, lr_plan={30: 0.0001}, wd=0.0005)
    nn.train(train_gen, test_gen, epochs=1, lr=0)
    #nn.test(test_gen)
    cfg.LOG.close_log()
    return

    # Fusing tests
    cfg.LOG.start_new_log(name='{}-{}-fuse'.format(arch, dataset))
    nn = NeuralNet(arch, dataset)
    nn.train(train_gen, test_gen, epochs=120, lr=0.01, lr_plan={30: 0.01, 60: 0.0001, 90: 0.00001}, wd=0.0005)
    cfg.LOG.close_log()
    return
    # I don't know
    cfg.LOG.start_new_log(name='{}-{}'.format(arch, dataset))
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    cfg.LOG.write('Statistics gathering for quantization')
    nn.test(test_gen, iterations=2, gather_stats=True, quantize=False, hw_sim=False)

    cfg.LOG.write('Inference')
    nn.test(test_gen, gather_stats=False, quantize=True, hw_sim=True)

    cfg.LOG.close_log()

    return


# -------------
# - GRAVEYARD -
# -------------

return self._masked_conv2d(x) if self._use_mask else self.conv(x)

# Can not continue without max/min mean statistics
assert (self.running_max_mean != 0 or self.running_min_mean != 0)

# Currently only supports unsigned uniform quantization
if torch.min(x) == 0:
    x_q, x_q_delta = uniform_quantization(x, self.running_max_mean, 8)
    self.stats['x_hist'] += torch.histc(x_q, bins=256, min=0, max=255)
else:
    # x_q, x_q_delta = uniform_symmetric_quantization(x, self.running_min_mean, self.running_max_mean, 8)
    raise NotImplementedError

weight_q, weight_q_delta = uniform_symmetric_quantization(self.conv.weight * self.conv_mask.weight.data,
                                                          torch.min(self.conv.weight.data),
                                                          torch.max(self.conv.weight.data), 8)

if self.conv.bias is None:
    bias_fp = None
else:
    bias_q, bias_q_delta = uniform_symmetric_quantization(self.conv.bias,
                                                          torch.min(self.conv.bias.data),
                                                          torch.max(self.conv.bias.data), 8)
    bias_fp = bias_q[None, :, None, None] * bias_q_delta

# Return convolution results before unfolding
if not self._unfold:
    return nn.functional.conv2d(x_q * x_q_delta,
                                weight_q * weight_q_delta,
                                bias=bias_fp,
                                stride=self.conv.stride[0],
                                padding=self.conv.padding[0])

# Unfolding
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
    self.sort_stats += (x_unf > 15).sum(dim=(0, 1))

# Data reorder
if self._reorder:
    _, sort_idx = self.sort_stats.sort()
    # _, sort_idx = (x_unf > 15).sum(dim=(0, 1)).sort()
    sort_idx[int(sort_idx.size(0) / 2):] = sort_idx[int(sort_idx.size(0) / 2):].__reversed__()
    x_unf = x_unf[:, :, sort_idx]
    w_unf = w_unf[sort_idx, :]

if not self._hw_sim:
    out_unf = x_unf.matmul(w_unf).transpose(1, 2)

    if bias_fp is None:
        bias_fp = 0

    out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
    out = out * x_q_delta * weight_q_delta + bias_fp
    return out

# Only supports two threads at the moment
# HW simulation
threads = 2

if (x_unf.size(2) % 2) != 0:  # TODO: should check for threads number
    x_unf = torch.cat((x_unf, torch.zeros(x_unf.size(0), x_unf.size(1), 1).cuda()), dim=2)
    w_unf = torch.cat((w_unf, torch.zeros(1, w_unf.size(1)).cuda()))

aaa = x_unf.reshape(x_unf.size(0), x_unf.size(1), threads, int(x_unf.size(2) / threads))[:, :, None, :, :] \
    .expand([-1, -1, w_unf.size(1), -1, -1]) \
    .contiguous()
bbb = w_unf.t().reshape(w_unf.size(1), threads, int(w_unf.size(0) / threads))[None, None, :, :, :] \
    .expand(x_unf.size(0), x_unf.size(1), -1, -1, -1)

with torch.no_grad():
    # Get a binary mask of the indices, within the threads, where more than one computations occur
    x_coll = (((aaa.detach() * bbb.detach()) != 0).sum(dim=3) == 2)
    x_coll = x_coll[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
    # And in addition, if collision occurs, whether an 8-bit activation was involved
    x_coll = x_coll * (aaa > 15)

# In case two 8-bit activations, we leave both numbers MSBs
qqq = (x_coll[:, :, :, 0, :] == True) & (x_coll[:, :, :, 1, :] == True)
aaa[:, :, :, 0, :] = torch.where(qqq == True,
                                 FloorSTE.apply((aaa[:, :, :, 0, :]) / 16) * 16,
                                 aaa[:, :, :, 0, :])
aaa[:, :, :, 1, :] = torch.where(qqq == True,
                                 FloorSTE.apply((aaa[:, :, :, 1, :]) / 16) * 16,
                                 aaa[:, :, :, 1, :])
self.stats['8b8b'] += qqq.sum().item()

# In case thread 0 has an 8-bit activation, thread 1 is cancelled
qqq = (x_coll[:, :, :, 0, :] == True) & (x_coll[:, :, :, 1, :] == False)
aaa[:, :, :, 1, :] = torch.where(qqq == True,
                                 torch.zeros_like(aaa[:, :, :, 1, :]),
                                 aaa[:, :, :, 1, :])
self.stats['8b4b'] += qqq.sum().item()

# In case thread 1 has an 8-bit activation, thread 0 is cancelled
qqq = (x_coll[:, :, :, 0, :] == False) & (x_coll[:, :, :, 1, :] == True)
aaa[:, :, :, 0, :] = torch.where(qqq == True,
                                 torch.zeros_like(aaa[:, :, :, 0, :]),
                                 aaa[:, :, :, 0, :])
self.stats['8b4b'] += qqq.sum().item()

qqq = (x_coll[:, :, :, 0, :] == False) & (x_coll[:, :, :, 1, :] == False)
self.stats['4b4b'] += qqq.sum().item()

# Finalizing multiplications
data_tensor = aaa * bbb
data_tensor = data_tensor.sum(dim=(3, 4))
out_unf = data_tensor.transpose(1, 2)

if bias_fp is None:
    bias_fp = 0
else:
    bias_fp = bias_q[None, :, None, None] * bias_q_delta

out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
out = out * x_q_delta * weight_q_delta + bias_fp  # bias_q[None, :, None, None] * bias_q_delta

self.stats['mse_error'] = self.stats['mse_error'] * self.stats['inputs']
self.stats['mse_error'] += torch.nn.functional.mse_loss(out, self.conv(x)).item()
self.stats['inputs'] += out.size(0)
self.stats['mse_error'] = self.stats['mse_error'] / self.stats['inputs']
self.stats['mac_count'] += out.size(0) * out.size(1) * out.size(2) * out.size(3) * \
                           self.conv.weight.size(1) * self.conv.weight.size(2) * self.conv.weight.size(3)

return out

# coll = (non_zero_elems == 4)[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
# aaa = torch.where((coll & (aaa > 15)) == True,
#                  FloorSTE.apply(aaa / 16) * 16,
#                  aaa)
# bbb = torch.where((coll & ((bbb > 7) | (bbb < -8))) == True,
#                  FloorSTE.apply(bbb / 8) * 8,
#                  bbb)

# coll = (non_zero_elems == 3)[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
# qqq = (((coll & (aaa > 15)) + (coll & ((bbb > 7) | (bbb < -8)))).sum(dim=3) > 1)[:, :, :, None, :].expand(
#    [-1, -1, -1, threads, -1])
# aaa = torch.where(qqq == True,
#                  FloorSTE.apply(aaa / 16) * 16,
#                  aaa)
# bbb = torch.where(qqq == True,
#                  FloorSTE.apply(bbb / 8) * 8,
#                  bbb)


# with torch.no_grad():
#    # Get a binary mask of the indices, within the threads, where more than one computations occur
#    x_coll = (((aaa.detach() * bbb.detach()) != 0).sum(dim=3) == 2)
#    x_coll = x_coll[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
#    # And in addition, if collision occurs, whether an 8-bit activation was involved
#    w_coll = x_coll * ((bbb > 7) | (bbb < -8))
#    x_coll = x_coll * (aaa > 15)

# qqq = (x_coll[:, :, :, 0, :] == False) & (x_coll[:, :, :, 1, :] == True) & (w_coll[:, :, :, 1, :] == True)
# aaa[:, :, :, 1, :] = torch.where(qqq == True,
#                                 FloorSTE.apply((aaa[:, :, :, 1, :]) / 16) * 16,
#                                 aaa[:, :, :, 1, :])
# self.stats['x1-4b_x2-8b_w2-8b'] += qqq.sum().item()

# qqq = (x_coll[:, :, :, 0, :] == True) & (x_coll[:, :, :, 1, :] == False) & (w_coll[:, :, :, 0, :] == True)
# aaa[:, :, :, 0, :] = torch.where(qqq == True,
#                                 FloorSTE.apply((aaa[:, :, :, 0, :]) / 16) * 16,
#                                 aaa[:, :, :, 0, :])
# self.stats['x1-8b_x2-4b_w1-8b'] += qqq.sum().item()

# qqq = (x_coll[:, :, :, 0, :] == True) & (x_coll[:, :, :, 1, :] == True) & (w_coll[:, :, :, 0, :] == False) & (w_coll[:, :, :, 1, :] == True)
# aaa[:, :, :, 1, :] = torch.where(qqq == True,
#                                 FloorSTE.apply((aaa[:, :, :, 1, :]) / 16) * 16,
#                                 aaa[:, :, :, 1, :])
# self.stats['x1-8b_x2-8b_w1-4b_w2-8b'] += qqq.sum().item()

# qqq = (x_coll[:, :, :, 0, :] == True) & (x_coll[:, :, :, 1, :] == True) & (w_coll[:, :, :, 0, :] == True) & (w_coll[:, :, :, 1, :] == False)
# aaa[:, :, :, 0, :] = torch.where(qqq == True,
#                                 FloorSTE.apply((aaa[:, :, :, 0, :]) / 16) * 16,
#                                 aaa[:, :, :, 0, :])
# self.stats['x1-8b_x2-8b_w1-8b_w2-4b'] += qqq.sum().item()

# qqq = (x_coll[:, :, :, 0, :] == True) & (x_coll[:, :, :, 1, :] == True) & (w_coll[:, :, :, 0, :] == True) & (w_coll[:, :, :, 1, :] == True)
# aaa[:, :, :, 0, :] = torch.where(qqq == True,
#                                 FloorSTE.apply((aaa[:, :, :, 0, :]) / 16) * 16,
#                                 aaa[:, :, :, 0, :])
# aaa[:, :, :, 1, :] = torch.where(qqq == True,
#                                 FloorSTE.apply((aaa[:, :, :, 1, :]) / 16) * 16,
#                                 aaa[:, :, :, 1, :])
# self.stats['x1-8b_x2-8b_x3-8b_x4-8b'] += qqq.sum().item()

# self.stats['total'] += qqq.numel()


if self._hw_mac == '2x4bx4b':
    # TODO: confirm implementation
    raise NotImplementedError
    # Skip this part and save time if threads < 2
    if threads == 2:
        coll = (non_zero_elems == 2)[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
        qqq = coll & (aaa > 15)
        aaa = torch.where(qqq,
                          trunc_func.apply(aaa / 16) * 16,
                          aaa)
        self.stats['Col2_A4'] += qqq.sum().item()

        qqq = coll & ((bbb > 7) | (bbb < -8))
        bbb = torch.where(qqq,
                          trunc_func.apply(bbb / 16) * 16,
                          bbb)
        self.stats['Col2_W4'] += qqq.sum().item()
    else:
        # Just to be sure
        assert (non_zero_elems == 2).sum().item() == 0

    coll = (non_zero_elems == 1)[:, :, :, None, :].expand([-1, -1, -1, threads, -1])
    if self._hw_type == 'wA':
        qqq = (coll & (aaa > 15)) & (coll & ((bbb > 7) | (bbb < -8)))
    elif self._hw_type == 'A':
        qqq = (coll & (aaa > 15))
    else:
        raise NotImplementedError

    aaa = torch.where(qqq == True,
                      trunc_func.apply(aaa / 16) * 16,
                      aaa)
    self.stats['Col1_A4'] += qqq.sum().item()
    self.stats['total'] += qqq.numel()

# Bias and variance correction
# orig = self.conv(x)
# var_correction = (orig.var() / out.var()) ** 0.5
# out = out * var_correction
# mean_correction = out.mean() - orig.mean()
# out = out - mean_correction