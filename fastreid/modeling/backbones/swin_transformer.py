import logging
import torch.nn as nn
import timm

from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


class SwinTransformer(nn.Module):
    def __init__(self, depth, pretrained=False):
        super().__init__()
        self.model = timm.create_model(depth, pretrained=pretrained)

    def forward(self, x):
        x = self.model.patch_embed(x)
        if self.model.absolute_pos_embed is not None:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        x = self.model.norm(x)  # B L C
        x = x.transpose(1, 2)  # B C 1
        return x


@BACKBONE_REGISTRY.register()
def build_swin_backbone(cfg):
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

    model = SwinTransformer(depth, pretrain)

    return model
