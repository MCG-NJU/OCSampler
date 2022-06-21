import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp

from ...utils import get_root_logger
from ..builder import BACKBONES
from torchvision import models
import torch


def get_torchvision_model(name, pretrained=True, requires_grad=False, truncate_modules=None):
    torchvision_models = models
    if "." in name:
        prefix, name = name.split(".")[0], name.split(".")[1]
        assert prefix in vars(torchvision_models).keys()
        torchvision_models = vars(torchvision_models)[prefix]
    assert name in vars(torchvision_models).keys()

    if name == "inception_v3":
        model = vars(torchvision_models)[name](pretrained=pretrained, aux_logits=False)
    else:
        model = vars(torchvision_models)[name](pretrained=pretrained)
    if truncate_modules is not None:
        model = torch.nn.Sequential(*list(model.children())[:truncate_modules])
    for param in model.parameters():
        param.requires_grad = requires_grad

    if not requires_grad:
        model.eval()

    return model


def get_base_model(name='torchvision.resnet50'):
    truncate_modules = -1
    if name is None:
        return None
    if "torchvision" in name.lower():
        model_name = name.split(".", 1)[-1]
        model = get_torchvision_model(
            name=model_name,
            pretrained=True,
            requires_grad=False,
            truncate_modules=truncate_modules,
        )
    else:
        raise Exception("couldn't find %s as a model name" % name)

    return model


@BACKBONES.register_module()
class ResNet50(nn.Module):

    def __init__(self, norm_eval=True, freeze_all=True, pretrained=None):
        super().__init__()
        self.model = get_base_model()
        self.norm_eval = norm_eval
        self.freeze_all = freeze_all
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        return self.model(x)

    def _freeze_all(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        for m in self.model.modules():
            if isinstance(m, nn.Module):
                for param in m.parameters():
                    param.requires_grad = False
                m.eval()

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
