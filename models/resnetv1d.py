"""ResNetV1bs, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring,dangerous-default-value
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['resnet101_v1d']

import mxnet as mx


class DeformabelConv2D(nn.HybridBlock):
    def __init__(self, *args, **kwargs):
        super(DeformabelConv2D, self).__init__()
        init = mx.init.Zero()
        init.set_verbosity(True)
        with self.name_scope():
            self.conv_offset = nn.Conv2D(channels=72, kernel_size=3, strides=1, dilation=2, padding=2,
                                         weight_initializer=init, bias_initializer=init, prefix="offset_")

            init = mx.init.Normal()
            init.set_verbosity(True)
            self.weight = self.params.get('weight', shape=(512, 512, 3, 3),
                                          init=init,
                                          allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight):
        offset = self.conv_offset(x)
        data = F.contrib.DeformableConvolution(name='fwd',
                                               data=x,
                                               offset=offset,
                                               num_filter=512, pad=(2, 2), kernel=(3, 3),
                                               num_deformable_group=4,
                                               stride=(1, 1), dilate=(2, 2), no_bias=True, weight=weight)
        return data


# Helpers
def _conv3x3(channels, stride, use_dcn):
    if use_dcn:
        return DeformabelConv2D(channels, kernel_size=3, strides=stride, padding=2, dilation=2,
                     use_bias=False)
    else:
        return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1, dilation=1,
                     use_bias=False)


class BottleneckV1b(HybridBlock):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs={}, last_gamma=False, use_dcn=False, **kwargs):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1,
                               use_bias=False)
        self.bn1 = norm_layer(**norm_kwargs)
        self.relu1 = nn.Activation('relu')
        self.conv2 = _conv3x3(channels=planes, stride=strides, use_dcn=use_dcn)
        self.bn2 = norm_layer(**norm_kwargs)
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, use_bias=False)
        if not last_gamma:
            self.bn3 = norm_layer(**norm_kwargs)
        else:
            self.bn3 = norm_layer(gamma_initializer='zeros', **norm_kwargs)
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out


class ResNetV1b(HybridBlock):
    """ Pre-trained ResNetV1b Model, which preduces the strides of 8
    featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, classes=1000, dilated=False, norm_layer=BatchNorm,
                 norm_kwargs={}, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__(prefix=name_prefix)
        self.norm_kwargs = norm_kwargs
        if use_global_stats:
            self.norm_kwargs['use_global_stats'] = True
        with self.name_scope():
            if not deep_stem:
                self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False)
            else:
                self.conv1 = nn.HybridSequential(prefix='conv1')
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(**norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(**norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width * 2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
            self.bn1 = norm_layer(**norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            if dilated:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, use_dcn=True)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False, use_dcn=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_' % stage_index)
            with downsample.name_scope():
                if avg_down:
                    if dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False))
                    downsample.add(norm_layer(**self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False))
                    downsample.add(norm_layer(**self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_' % stage_index)
        with layers.name_scope():
            if dilation in (1, 2):
                layers.add(block(planes, strides, dilation=1,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma, use_dcn=use_dcn))
            elif dilation == 4:
                layers.add(block(planes, strides, dilation=2,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma, use_dcn=use_dcn))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(planes, dilation=dilation,
                                 previous_dilation=dilation, norm_layer=norm_layer,
                                 norm_kwargs=self.norm_kwargs, last_gamma=last_gamma, use_dcn=use_dcn))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


def resnet101_v1d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        model.load_parameters(get_model_file('resnet%d_v%dd' % (101, 1),
                                             tag=pretrained, root=root), allow_missing=True, ignore_extra=True, ctx=ctx)
    return model
