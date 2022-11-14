# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        return_layers = {'layer1': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.frozen_batch_norm1 = FrozenBatchNorm2d(512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=1)
        self.frozen_batch_norm2 = FrozenBatchNorm2d(512)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=7, stride=2, padding=1)
        self.frozen_batch_norm3 = FrozenBatchNorm2d(1024)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=5, stride=2, padding=1)
        self.frozen_batch_norm4 = FrozenBatchNorm2d(1024)
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=1)
        self.frozen_batch_norm5 = FrozenBatchNorm2d(1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=2, padding=1)
        self.frozen_batch_norm6 = FrozenBatchNorm2d(2048)
        self.conv7 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=5, stride=1, padding=1)
        self.frozen_batch_norm7 = FrozenBatchNorm2d(2048)
        self.relu = nn.ReLU()
        self.num_channels = num_channels # 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for key, value in xs.items(): #add line
            xs[key] = self.conv1(value)
            xs[key] = self.frozen_batch_norm1(xs[key])
            xs[key] = self.relu(xs[key])
            xs[key] = self.conv2(xs[key])
            xs[key] = self.frozen_batch_norm2(xs[key])
            xs[key] = self.relu(xs[key])
            xs[key] = self.conv3(xs[key])
            xs[key] = self.frozen_batch_norm3(xs[key])
            xs[key] = self.relu(xs[key])
            xs[key] = self.conv4(xs[key])
            xs[key] = self.frozen_batch_norm4(xs[key])
            xs[key] = self.relu(xs[key])
            xs[key] = self.conv5(xs[key])
            xs[key] = self.frozen_batch_norm5(xs[key])
            xs[key] = self.relu(xs[key])
            xs[key] = self.conv6(xs[key])
            xs[key] = self.frozen_batch_norm6(xs[key])
            xs[key] = self.relu(xs[key])
            xs[key] = self.conv7(xs[key])
            xs[key] = self.frozen_batch_norm7(xs[key])
            xs[key] = self.relu(xs[key])
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
