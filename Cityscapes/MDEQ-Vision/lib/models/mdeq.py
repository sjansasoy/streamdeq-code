from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

sys.path.append("lib/models")
from mdeq_core import MDEQNet

sys.path.append("../")
from lib.layer_utils import conv3x3, list2vec, vec2list

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class MDEQClsNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQClsNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
        self.classifier = nn.Linear(self.final_chansize, self.num_classes)

    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        head_block = Bottleneck
        d_model = self.init_chansize
        head_channels = self.head_channels

        # Increasing the number of channels on each resolution when doing classification.
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
                                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        # Final FC layers
        final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels) - 1] * head_block.expansion,
                                              self.final_chansize, kernel_size=1),
                                    nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def predict(self, y_list):
        """
        Given outputs at multiple resolutions, predict the class of the image
        """
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        # Pool to a 1x1 vector (if needed)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y

    def forward(self, x, train_step=0, **kwargs):
        y_list, jac_loss, sradius = self._forward(x, train_step, **kwargs)
        return self.predict(y_list), jac_loss, sradius

    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MDEQSegNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQSegNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)

        # Last layer
        last_inp_channels = np.int(np.sum(self.num_channels))
        self.last_layer = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1),
                                        nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(last_inp_channels, cfg.DATASET.NUM_CLASSES,
                                                  cfg.MODEL.EXTRA.FINAL_CONV_KERNEL,
                                                  stride=1, padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0))

        # Temporal memory used by the original StreamDEQ warm-start.
        # It stores the most recent multiscale equilibrium representation.
        self.prev_outs = None

        # ML project: history buffer for stale initialization experiments.
        # This will allow us to initialize the solver from an older state, e.g. z_{t-2}
        # instead of always using the immediately previous state z_{t-1}.
        self.prev_outs_history = []

    def segment(self, y):
        """
        Given outputs at multiple resolutions, segment the feature map by predicting the class of each pixel
        """
        # Segmentation Head
        y0_h, y0_w = y[0].size(2), y[0].size(3)
        all_res = [y[0]]
        for i in range(1, self.num_branches):
            all_res.append(F.interpolate(y[i], size=(y0_h, y0_w), mode='bilinear', align_corners=True))

        y = torch.cat(all_res, dim=1)
        all_res = None
        y = self.last_layer(y)
        return y

    # ML project: helper function to build a partial initialization state by reusing only selected scales from the previous frame.
    def _build_partial_init_state(self, prev_outs, partial_init_mode):
        """
        Build a multiscale initialization state by reusing only selected
        scales from the previous frame and setting the remaining scales to zero.

        The order of scales was diagnosed as:
            scale 0: high resolution / fine      (1, 88, 256, 512)
            scale 1: mid-high resolution         (1, 176, 128, 256)
            scale 2: mid-low resolution          (1, 352, 64, 128)
            scale 3: low resolution / coarse     (1, 704, 32, 64)

        Therefore:
            fine scales   = [0, 1]
            coarse scales = [2, 3]
        """
        if prev_outs is None:
            # At the beginning of a sequence there is no previous state.
            # Returning None lets _forward use its internal zero initialization.
            return None

        if not isinstance(prev_outs, (list, tuple)):
            raise TypeError(
                f'Expected prev_outs to be a list or tuple, got {type(prev_outs)}'
            )

        if len(prev_outs) != 4:
            raise ValueError(
                f'Expected 4 multiscale states in prev_outs, got {len(prev_outs)}'
            )

        # Start with zeros at every scale.
        init_state = [torch.zeros_like(z) for z in prev_outs]

        if partial_init_mode == 'coarse_previous_fine_zero':
            # Reuse low-resolution semantic scales.
            reuse_indices = [2, 3]

        elif partial_init_mode == 'fine_previous_coarse_zero':
            # Reuse high-resolution spatial/detail scales.
            reuse_indices = [0, 1]


        elif partial_init_mode == 'only_high_resolution_previous':
            # Reuse only the finest/highest-resolution scale.
            reuse_indices = [0]

        elif partial_init_mode == 'only_low_resolution_previous':
            # Reuse only the coarsest/lowest-resolution scale.
            reuse_indices = [3]

        elif partial_init_mode == 'only_scale1_previous':
            reuse_indices = [1]

        elif partial_init_mode == 'only_scale2_previous':
            reuse_indices = [2]

        elif partial_init_mode == 'scale1_scale3_previous':
            reuse_indices = [1, 3]

        elif partial_init_mode == 'scale0_scale2_previous':
            # Reuse scale 0 and scale 2.
            reuse_indices = [0, 2]

        elif partial_init_mode == 'scale1_scale2_previous':
            # Reuse the two intermediate scales.
            reuse_indices = [1, 2]

        elif partial_init_mode == 'scale0_scale3_previous':
            # Reuse the finest scale 0 and the coarsest scale 3.
            reuse_indices = [0, 3]

        elif partial_init_mode == 'scale1_scale2_scale3_previous':
            # Reuse all scales except the finest/highest-resolution scale 0.
            reuse_indices = [1, 2, 3]

        elif partial_init_mode == 'drop_scale0_previous':
            reuse_indices = [1, 2, 3]

        elif partial_init_mode == 'drop_scale1_previous':
            reuse_indices = [0, 2, 3]

        elif partial_init_mode == 'drop_scale2_previous':
            reuse_indices = [0, 1, 3]

        elif partial_init_mode == 'drop_scale3_previous':
            reuse_indices = [0, 1, 2]

        elif partial_init_mode == 'all_previous':
            # Equivalent to the original StreamDEQ warm-start.
            reuse_indices = [0, 1, 2, 3]

        elif partial_init_mode == 'all_zero':
            # Explicit all-zero multiscale initialization.
            reuse_indices = []

        else:
            raise ValueError(f'Unknown partial_init_mode: {partial_init_mode}')

        for idx in reuse_indices:
            init_state[idx] = prev_outs[idx]

        return init_state

    def _select_init_state(self, init_mode, partial_init_mode='coarse_previous_fine_zero',
                        stale_k=2):
        """
        Select the initialization state passed to the DEQ forward solver.
        """
        if init_mode == 'previous':
            # Original StreamDEQ behavior.
            return self.prev_outs

        elif init_mode == 'zero':
            # _forward interprets None as zero initialization.
            return None

        elif init_mode == 'stale':
            # Use an older state if available; otherwise fall back to previous.
            if len(self.prev_outs_history) >= stale_k:
                return self.prev_outs_history[-stale_k]
            return self.prev_outs

        elif init_mode == 'partial':
            # New scale-aware initialization mode.
            return self._build_partial_init_state(
                self.prev_outs,
                partial_init_mode
            )

        else:
            raise ValueError(f'Unknown init_mode: {init_mode}')

    def forward(self, x, train_step=0, **kwargs):
        mode = kwargs.get('mode', 'baseline')

        # Initialization mode for the DEQ solver in streaming inference.
        # - "previous": original StreamDEQ behavior. The previous frame's
        #   equilibrium representation is used as the initial state.
        # - "zero": disables temporal warm-start by forcing the solver to
        #   start from zero/None even when running in stream mode.
        #
        # This option is useful for controlled experiments on the role of
        # initialization without changing the backbone or retraining the model.
        init_mode = kwargs.get('init_mode', 'previous')

        if mode == 'baseline':
            # Baseline MDEQ processes each frame independently.
            # Passing None makes _forward create a zero initialization internally.
            y, jac_loss, sradius = self._forward([x, None], train_step, **kwargs)
            return self.segment(y), jac_loss, sradius
        elif mode == 'unroll':
            kwargs.update({'deq_mode': False})
            y, jac_loss, sradius = self._forward([x, None], train_step, **kwargs)
            return self.segment(y), jac_loss, sradius
        elif mode != 'stream':
            raise ValueError('Mode is not defined.')

        # For streaming inference, we need to decide what initialization to pass to the DEQ solver.
        partial_init_mode = kwargs.get(
            'partial_init_mode',
            'coarse_previous_fine_zero'
        )
        stale_k = kwargs.get('stale_k', 2)

        init_state = self._select_init_state(
            init_mode=init_mode,
            partial_init_mode=partial_init_mode,
            stale_k=stale_k
        )

        y, jac_loss, sradius = self._forward([x, init_state], train_step, **kwargs)

        # ------------------------------------------------------------
        # Temporary diagnostic block: inspect multiscale state structure
        # ------------------------------------------------------------
        if not hasattr(self, "_printed_prevouts_diagnostic"):
            self._printed_prevouts_diagnostic = False

        if not self._printed_prevouts_diagnostic:
            print("\n[DEBUG prev_outs diagnostic]")
            print("mode:", mode)
            print("init_mode:", init_mode)
            print("type(y):", type(y))

            if isinstance(y, (list, tuple)):
                print("number of scales in y:", len(y))
                for i, yi in enumerate(y):
                    print(
                        f"scale {i}: shape={tuple(yi.shape)}, "
                        f"dtype={yi.dtype}, device={yi.device}"
                    )
            else:
                print("y is not a list or tuple.")
                if hasattr(y, "shape"):
                    print("y shape:", tuple(y.shape))

            if self.prev_outs is None:
                print("self.prev_outs is currently None")
            else:
                print("type(self.prev_outs):", type(self.prev_outs))
                if isinstance(self.prev_outs, (list, tuple)):
                    print("number of scales in self.prev_outs:", len(self.prev_outs))
                    for i, zi in enumerate(self.prev_outs):
                        print(
                            f"prev scale {i}: shape={tuple(zi.shape)}, "
                            f"dtype={zi.dtype}, device={zi.device}"
                        )

            self._printed_prevouts_diagnostic = True
        # ------------------------------------------------------------

        # Always store the current representation so that future frames can use it
        # when init_mode="previous". This keeps the original temporal memory logic.
        current_state = y.copy()
        self.prev_outs = current_state

        # ML project: keep a short history of previous multiscale states.
        # This buffer will be used by INIT_MODE="stale" to initialize the solver from
        # an older state instead of the immediately previous one.
        self.prev_outs_history.append(current_state)

        # Keep the buffer bounded to avoid storing unnecessary old states.
        # A length of 10 is enough for the first stale-k experiments.
        max_history_len = 10
        if len(self.prev_outs_history) > max_history_len:
            self.prev_outs_history.pop(0)

        return self.segment(y), jac_loss, sradius

    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info(f'=> init weights from normal distribution. PRETRAINED={pretrained}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()

            # Just verification...
            diff_modules = set()
            for k in pretrained_dict.keys():
                if k not in model_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In ImageNet MDEQ but not Cityscapes MDEQ: {sorted(list(diff_modules))}", "red"))
            diff_modules = set()
            for k in model_dict.keys():
                if k not in pretrained_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In Cityscapes MDEQ but not ImageNet MDEQ: {sorted(list(diff_modules))}", "green"))

            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def reset_prev_outs(self):
        # Reset temporal memory at the end of each sequence.
        self.prev_outs = None

        # ML project: also reset the history buffer used by stale initialization.
        # This prevents states from one clip from being reused in the next clip.
        self.prev_outs_history = []


def get_cls_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQClsNet(config, **kwargs)
    model.init_weights()
    return model


def get_seg_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQSegNet(config, **kwargs)
    model.init_weights(config.MODEL.PRETRAINED)
    return model
