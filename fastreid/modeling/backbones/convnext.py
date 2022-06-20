import logging
import torch.nn as nn
import timm

from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


class ConvNeXt(nn.Module):
    def __init__(self, depth, pretrained=False):
        super().__init__()
        self.model = timm.create_model(depth, pretrained=pretrained)
        self.model.stages[3].downsample[1].stride = (1, 1)

    def forward(self, x):
        x = self.model.forward_features(x)
        return x


@BACKBONE_REGISTRY.register()
def build_convnext_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    model = ConvNeXt(depth, pretrain)

    return model
