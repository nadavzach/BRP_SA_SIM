import argparse
import os
import matplotlib
import numpy as np
import torch
import sys
import Config as cfg
from NeuralNet import NeuralNet

#import cpy_smt_sa
#from smt_sa.smt_sa import SMTSA
#from util.torchsummary import summary

#matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='Gil Shomron, gilsho@campus.technion.ac.il',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = ['alexnet-cifar100', 'alexnet-imagenet',
               'resnet18-imagenet',
               'resnet50-imagenet',
               'googlenet-imagenet',
               'densenet-imagenet']

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=True,
                    help='model architectures and datasets:\n ' + ' | '.join(model_names))
parser.add_argument('--action', choices=['PRUNE', 'QUANTIZE', 'INFERENCE'], required=True,
                    help='PRUNE: magnitude pruning\n'
                         'QUANTIZE: symmetric min-max uniform quantization\n'
                         'INFERENCE: either regular inference or hardware simulated inference')
parser.add_argument('--desc')
parser.add_argument('--chkp', default=None, metavar='PATH',
                    help='model checkpoint')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--x_bits', default=8, type=int, metavar='N',
                    help='activations quantization bits')
parser.add_argument('--w_bits', default=8, type=int, metavar='N',
                    help='weights quantization bits')
parser.add_argument('--hw_sim', action='store_true',
                    help='toggle SMT-SA simulation on inference')
parser.add_argument('--mac', choices=['2x4bx4b', '2x4bx8b'], default='2x4bx8b',
                    help='number of available HW MAC units')
parser.add_argument('--threads', choices=['4', '2', '1'], default=4,
                    help='number of default threads')
parser.add_argument('--reorder', choices=['OFF', 'STATS', 'IDEAL', 'WEIGHTS'], default='STATS',
                    help='OFF: disable data reordering\n'
                         'STATS: based on gathered statistics\n'
                         'IDEAL: based on each batch statistics\n'
                         'WEIGHTS: reorder weights')
parser.add_argument('--floor', action='store_true',
                    help='floor on truncation (default: round)')
parser.add_argument('--disable_sparsity', action='store_true',
                    help='do not exploit sparsity')
parser.add_argument('--hw_arch', choices=['wA', 'A', 'A0', 'aW', 'W', 'W0'], default='WA',
                    help='wA: A are trimmed, considering A and W bitwidth\n'
                         'A: A are trimmed, considering A bitwidth\n'
                         'A0: A are trimmed\n'
                         'aW: W are trimmed, considering A and W bitwidth\n'
                         'W: W are trimmed, considering W bitwidth\n'
                         'W0: W are trimmed\n')
parser.add_argument('--layers_t2', nargs='+', default=None,
                    help='list of layers that will run with two threads')
parser.add_argument('--layers_t1', nargs='+', default=None,
                    help='list of layers that will run with one threads')
parser.add_argument('--t1_analysis', action='store_true',
                    help='measure one thread MAC bit-width distribution, set -threads=1')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')


def prune_network(arch, dataset, train_gen, test_gen, model_chkp=None, desc=None):
    # Learning both Weights and Connections for Efficient Neural Networks
    name_str = '{}-{}_prune'.format(arch, dataset)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('desc={}'.format(desc))
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    # Only quantizing CONV layers; therefore, disable FC layers gradients
    for name, param in nn.model.named_parameters():
        if 'fc' in name:
            param.requires_grad = False

    nn.model.bypass_all()
    nn.model.prune_all()

    for prune_ratio in [10, 20, 30, 40, 50, 60]:
        nn.best_top1_acc = 0
        nn.next_train_epoch = 0
        for l in nn.model.prune_list:
            th = np.quantile(l.conv.weight.data.abs().detach().cpu().numpy(), prune_ratio/100)
            l.conv_mask.weight.data = torch.gt(l.conv.weight.data.abs(), th).float()

        if arch == 'alexnet' and dataset == 'imagenet':
            nn.train(train_gen, test_gen, epochs=5, lr=0.00001, wd=0.0005, desc='wprune-{}'.format(prune_ratio))
        if arch == 'resnet18' and dataset == 'imagenet':
            nn.train(train_gen, test_gen, epochs=5, lr=0.0001, wd=0.0001, desc='wprune-{}'.format(prune_ratio))
        if arch == 'resnet50' and dataset == 'imagenet':
            nn.train(train_gen, test_gen, epochs=2, lr=0.0001, wd=0.0001, desc='wprune-{}'.format(prune_ratio))
        if arch == 'densenet' and dataset == 'imagenet':
            nn.train(train_gen, test_gen, epochs=2, lr=0.0001, wd=0.0001, desc='wprune-{}'.format(prune_ratio))

    cfg.LOG.close_log()
    return


def quantize_network(arch, dataset, train_gen, test_gen, model_chkp=None, only_stats=False,
                     x_bits=8, w_bits=8, desc=None):
    name_str = '{}-{}_quantize_x-{}_w-{}'.format(arch, dataset, x_bits, w_bits)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('desc={}, only_stats={}, x_bits={}, w_bits={}'.format(desc, only_stats, x_bits, w_bits))
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    nn.model.disable_fc_grad()
    nn.model.bypass_all()
    nn.model.prune_all()
    nn.model.quantize_all()
    nn.model.set_quantization_bits(x_bits, w_bits)

    nn.best_top1_acc = 0
    nn.next_train_epoch = 0

    # Quantize with unfold to gather reorder stats
    for l in nn.model.unfold_list:
        l._unfold = True

    if only_stats:
        nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)
    else:
        nn.next_train_epoch = 0
        nn.best_top1_acc = 0

        if arch == 'alexnet' and dataset == 'imagenet':
            nn.train(train_gen, test_gen, epochs=5, lr=0.000001, wd=5e-4)
        if arch == 'resnet18' and dataset == 'imagenet':
            nn.train(train_gen, test_gen, epochs=20, lr=0.0001, wd=0.0001)

    nn.print_stats()

    cfg.LOG.close_log()
    return


def inference(arch, dataset, test_gen, model_chkp, hw_sim=True, reorder='STATS', x_bits=8, w_bits=8, threads=2,
              is_round=False, hw_type='WA', sparsity=True, mac='2x4bx8b', layers_t2=None, layers_t1=None,
              t1_analysis=False, desc=None):
    layers_t1 = [] if layers_t1 is None else layers_t1
    layers_t2 = [] if layers_t2 is None else layers_t2
    name_str = '{}-{}_inference_x-{}_w-{}_{}_{}_spr-{}_reorder-{}_mac-{}_T{}'.format(arch, dataset, x_bits, w_bits,
                                                                          'round' if is_round else 'floor', hw_type,
                                                                          sparsity, reorder, mac, threads)
    name_str = name_str + '_T2-{}'.format('-'.join(map(str, layers_t2))) if layers_t2 != [] else name_str
    name_str = name_str + '_T1-{}'.format('-'.join(map(str, layers_t1))) if layers_t1 != [] else name_str
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('desc={}, hw_sim={}, reorder={}, x_bits={}, w_bits={}, is_round={}, hw_type={}, mac={}'
                  .format(desc, hw_sim, reorder, x_bits, w_bits, is_round, hw_type, mac))
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    nn.model.bypass_all()
    nn.model.prune_all()
    if model_chkp is not None:
        nn.model.quantize_all()
        nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_trunc_round() if is_round else nn.model.set_trunc_floor()
    nn.model.set_sparsity(sparsity)
    nn.model.set_hw_type(hw_type)

    for l_idx, l in enumerate(nn.model.unfold_list):
        l._unfold = hw_sim
        l._hw_sim = hw_sim
        l._reorder = reorder
        l._hw_mac = mac

        # Configure thread number
        if l_idx in layers_t2:
            l._threads = 2
        elif l_idx in layers_t1:
            l._threads = 1
        else:
            l._threads = threads

        l._t1_analysis = t1_analysis

    cfg.LOG.write_title('Configurations')
    nn.print_stats(only_cfg=True)

    cfg.LOG.write_title('Start Test')
    nn.test(test_gen, iterations=None)

    # Print statistics and save to dump
    cfg.LOG.write_title('Statistics')
    nn.print_stats()
    nn.print_stats(t1_analysis=t1_analysis)

    return


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    args.layers_t1 = list(map(int, args.layers_t1)) if args.layers_t1 is not None else []
    args.layers_t2 = list(map(int, args.layers_t2)) if args.layers_t2 is not None else []
    cfg.BATCH_SIZE = args.batch_size
    cfg.USER_CMD = ' '.join(sys.argv)

    arch = args.arch.split('-')[0]
    dataset = args.arch.split('-')[1]
    dataset_ = cfg.get_dataset(dataset)

    test_gen, _ = dataset_.testset(batch_size=args.batch_size)
    (train_gen, _), (_, _) = dataset_.trainset(batch_size=args.batch_size, max_samples=None, random_seed=16)

    if args.chkp is None:
        model_chkp = None
    else:
        model_chkp = cfg.RESULTS_DIR + '/' + args.chkp

    if args.action == 'PRUNE':
        prune_network(arch, dataset, train_gen, test_gen)

    elif args.action == 'QUANTIZE':
        quantize_network(arch, dataset, train_gen, test_gen,
                         model_chkp=model_chkp,
                         only_stats=True, x_bits=args.x_bits, w_bits=args.w_bits, desc=args.desc)

    elif args.action == 'INFERENCE':
        inference(arch, dataset, test_gen,
                  model_chkp=model_chkp,
                  hw_sim=args.hw_sim, reorder=args.reorder,
                  x_bits=args.x_bits, w_bits=args.w_bits, is_round=(not args.floor), hw_type=args.hw_arch,
                  threads=int(args.threads), sparsity=(not args.disable_sparsity), mac=args.mac,
                  layers_t2=args.layers_t2, layers_t1=args.layers_t1, t1_analysis=args.t1_analysis, desc=args.desc)

    return


if __name__ == '__main__':
    main()
