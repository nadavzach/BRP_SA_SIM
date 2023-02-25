import os
from Logger import Logger
from Datasets import Datasets
from models.alexnet_cifar import alexnet_cifar100
from models.alexnet_imagenet import alexnet
from models.resnet_imagenet import resnet18, resnet50
from models.googlenet import googlenet
from models.densenet import densenet121

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

MODELS = {'alexnet_cifar100': alexnet_cifar100,
          'alexnet_imagenet': alexnet,
          'resnet18_imagenet': resnet18,
          'resnet50_imagenet': resnet50,
          'googlenet_imagenet': googlenet,
          'densenet_imagenet': densenet121}

BATCH_SIZE = 128


# ------------------------------------------------
#                   Directories
# ------------------------------------------------
CHECKPOINT_DIR = os.path.join(basedir, 'checkpoint')
RESULTS_DIR = os.path.join(basedir, 'results')
DATASET_DIR = os.path.join(basedir, 'datasets')
DATASET_DIR_IMAGENET = '/mnt/p3700/gilsho/ilsvrc2012'


# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
LOG = Logger()


def get_model_chkp(arch, dataset):
    if arch == 'alexnet' and dataset == 'cifar100':
        filename = '_alexnet_cifar100_epoch-90_top1-67.28.pth'
        return '{}/{}'.format(RESULTS_DIR, filename)
    # For ImageNet PyTorch pretrained models are used
    elif dataset == 'imagenet':
        return None
    else:
        raise NotImplementedError


def get_dataset(dataset):
    if dataset == 'cifar100':
        return Datasets.get('CIFAR100', DATASET_DIR)
    elif dataset == 'imagenet':
        return Datasets.get('ImageNet', DATASET_DIR_IMAGENET)
    else:
        raise NotImplementedError
