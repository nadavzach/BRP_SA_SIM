from torch import nn
from .utils import load_state_dict_from_url
from QuantConv2d import QuantConv2d, UnfoldConv2d
from SimModel import SimModel


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            UnfoldConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            UnfoldConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(SimModel):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #        if m.bias is not None:
        #            nn.init.zeros_(m.bias)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.ones_(m.weight)
        #        nn.init.zeros_(m.bias)
        #    elif isinstance(m, nn.Linear):
        #        nn.init.normal_(m.weight, 0, 0.01)
        #        nn.init.zeros_(m.bias)

        self.update_unfold_list()
        self.prune_list = self.unfold_list.copy()
        self.prune_list.pop(0)  # Remove first layer
        self.quant_list = self.prune_list.copy()

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)

        state_dict['features.0.0.conv.weight'] = state_dict.pop('features.0.0.weight')
        state_dict['features.1.conv.0.0.conv.weight'] = state_dict.pop('features.1.conv.0.0.weight')
        state_dict['features.1.conv.1.conv.weight'] = state_dict.pop('features.1.conv.1.weight')
        state_dict['features.2.conv.0.0.conv.weight'] = state_dict.pop('features.2.conv.0.0.weight')
        state_dict['features.2.conv.1.0.conv.weight'] = state_dict.pop('features.2.conv.1.0.weight')
        state_dict['features.2.conv.2.conv.weight'] = state_dict.pop('features.2.conv.2.weight')
        state_dict['features.3.conv.0.0.conv.weight'] = state_dict.pop('features.3.conv.0.0.weight')
        state_dict['features.3.conv.1.0.conv.weight'] = state_dict.pop('features.3.conv.1.0.weight')
        state_dict['features.3.conv.2.conv.weight'] = state_dict.pop('features.3.conv.2.weight')
        state_dict['features.4.conv.0.0.conv.weight'] = state_dict.pop('features.4.conv.0.0.weight')
        state_dict['features.4.conv.1.0.conv.weight'] = state_dict.pop('features.4.conv.1.0.weight')
        state_dict['features.4.conv.2.conv.weight'] = state_dict.pop('features.4.conv.2.weight')
        state_dict['features.5.conv.0.0.conv.weight'] = state_dict.pop('features.5.conv.0.0.weight')
        state_dict['features.5.conv.1.0.conv.weight'] = state_dict.pop('features.5.conv.1.0.weight')
        state_dict['features.5.conv.2.conv.weight'] = state_dict.pop('features.5.conv.2.weight')
        state_dict['features.6.conv.0.0.conv.weight'] = state_dict.pop('features.6.conv.0.0.weight')
        state_dict['features.6.conv.1.0.conv.weight'] = state_dict.pop('features.6.conv.1.0.weight')
        state_dict['features.6.conv.2.conv.weight'] = state_dict.pop('features.6.conv.2.weight')
        state_dict['features.7.conv.0.0.conv.weight'] = state_dict.pop('features.7.conv.0.0.weight')
        state_dict['features.7.conv.1.0.conv.weight'] = state_dict.pop('features.7.conv.1.0.weight')
        state_dict['features.7.conv.2.conv.weight'] = state_dict.pop('features.7.conv.2.weight')
        state_dict['features.8.conv.0.0.conv.weight'] = state_dict.pop('features.8.conv.0.0.weight')
        state_dict['features.8.conv.1.0.conv.weight'] = state_dict.pop('features.8.conv.1.0.weight')
        state_dict['features.8.conv.2.conv.weight'] = state_dict.pop('features.8.conv.2.weight')
        state_dict['features.9.conv.0.0.conv.weight'] = state_dict.pop('features.9.conv.0.0.weight')
        state_dict['features.9.conv.1.0.conv.weight'] = state_dict.pop('features.9.conv.1.0.weight')
        state_dict['features.9.conv.2.conv.weight'] = state_dict.pop('features.9.conv.2.weight')
        state_dict['features.10.conv.0.0.conv.weight'] = state_dict.pop('features.10.conv.0.0.weight')
        state_dict['features.10.conv.1.0.conv.weight'] = state_dict.pop('features.10.conv.1.0.weight')
        state_dict['features.10.conv.2.conv.weight'] = state_dict.pop('features.10.conv.2.weight')
        state_dict['features.11.conv.0.0.conv.weight'] = state_dict.pop('features.11.conv.0.0.weight')
        state_dict['features.11.conv.1.0.conv.weight'] = state_dict.pop('features.11.conv.1.0.weight')
        state_dict['features.11.conv.2.conv.weight'] = state_dict.pop('features.11.conv.2.weight')
        state_dict['features.12.conv.0.0.conv.weight'] = state_dict.pop('features.12.conv.0.0.weight')
        state_dict['features.12.conv.1.0.conv.weight'] = state_dict.pop('features.12.conv.1.0.weight')
        state_dict['features.12.conv.2.conv.weight'] = state_dict.pop('features.12.conv.2.weight')
        state_dict['features.13.conv.0.0.conv.weight'] = state_dict.pop('features.13.conv.0.0.weight')
        state_dict['features.13.conv.1.0.conv.weight'] = state_dict.pop('features.13.conv.1.0.weight')
        state_dict['features.13.conv.2.conv.weight'] = state_dict.pop('features.13.conv.2.weight')
        state_dict['features.14.conv.0.0.conv.weight'] = state_dict.pop('features.14.conv.0.0.weight')
        state_dict['features.14.conv.1.0.conv.weight'] = state_dict.pop('features.14.conv.1.0.weight')
        state_dict['features.14.conv.2.conv.weight'] = state_dict.pop('features.14.conv.2.weight')
        state_dict['features.15.conv.0.0.conv.weight'] = state_dict.pop('features.15.conv.0.0.weight')
        state_dict['features.15.conv.1.0.conv.weight'] = state_dict.pop('features.15.conv.1.0.weight')
        state_dict['features.15.conv.2.conv.weight'] = state_dict.pop('features.15.conv.2.weight')
        state_dict['features.16.conv.0.0.conv.weight'] = state_dict.pop('features.16.conv.0.0.weight')
        state_dict['features.16.conv.1.0.conv.weight'] = state_dict.pop('features.16.conv.1.0.weight')
        state_dict['features.16.conv.2.conv.weight'] = state_dict.pop('features.16.conv.2.weight')
        state_dict['features.17.conv.0.0.conv.weight'] = state_dict.pop('features.17.conv.0.0.weight')
        state_dict['features.17.conv.1.0.conv.weight'] = state_dict.pop('features.17.conv.1.0.weight')
        state_dict['features.17.conv.2.conv.weight'] = state_dict.pop('features.17.conv.2.weight')
        state_dict['features.18.0.conv.weight'] = state_dict.pop('features.18.0.weight')

        model.load_state_dict(state_dict, strict=False)
    return model
