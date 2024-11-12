# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .network_swinir import SwinIR

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=512, num_classes=1, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.swin_unet = SwinIR(img_size=512,
                                patch_size=64,
                                in_chans=1,
                                window_size=8,
                                depths=[1, 1, 1, 1],
                                upscale=1,
                                num_classes=1)

    def forward(self, x):
        #if x.size()[1] == 1:
        #    x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

 
