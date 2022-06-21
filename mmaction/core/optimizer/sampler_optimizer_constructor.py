import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd


@OPTIMIZER_BUILDERS.register_module()
class SamplerOptimizerConstructor(DefaultOptimizerConstructor):

    def add_params(self, params, model):

        sampler_fc_multiplier = self.paramwise_cfg.get('sampler_fc_multiplier', 5)
        sampler_conv_multiplier = self.paramwise_cfg.get('sampler_conv_multiplier', 1)
        sampler_bn_multiplier = self.paramwise_cfg.get('sampler_bn_multiplier', 1)

        sampler_fc_weight = []
        sampler_fc_bias = []
        sampler_conv_weight = []
        sampler_conv_bias = []
        sampler_bn = []

        normal_weight = []
        normal_bias = []
        normal_bn = []

        for name, module in model.named_modules():
            if isinstance(module, _ConvNd):
                m_params = list(m.parameters())
                if 'sampler' in name:
                    sampler_conv_weight.append(m_params[0])
                    if len(m_params) == 2:
                        sampler_conv_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m, torch.nn.Linear):
                m_params = list(m.parameters())
                if 'sampler' in name:
                    sampler_fc_weight.append(m_params[0])
                    if len(m_params) == 2:
                        sampler_fc_bias.append(m_params[1])
                else:
                    normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        normal_bias.append(m_params[1])
            elif isinstance(m,
                            (_BatchNorm, SyncBatchNorm, torch.nn.GroupNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        if 'sampler' in name:
                            sampler_bn.append(param)
                        else:
                            normal_bn.append(param)
            elif len(module._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(f'New atomic module type: {type(m)}. '
                                     'Need to give it a learning policy')

        params.append(
            {'params': sampler_conv_weight, 'lr': self.base_lr * sampler_conv_multiplier, 'weight_decay': self.base_wd})
        params.append(
            {'params': sampler_conv_bias, 'lr': self.base_lr * sampler_conv_multiplier * 2, 'weight_decay': 0})
        params.append(
            {'params': sampler_bn, 'lr': self.base_lr * sampler_bn_multiplier, 'weight_decay': 0})
        params.append(
            {'params': sampler_fc_weight, 'lr': self.base_lr * sampler_fc_multiplier, 'weight_decay': self.base_wd})
        params.append(
            {'params': sampler_fc_bias, 'lr': self.base_lr * sampler_fc_multiplier * 2, 'weight_decay': 0})

        params.append(
            {'params': normal_weight, 'lr': self.base_lr, 'weight_decay': self.base_wd})
        params.append(
            {'params': normal_bias, 'lr': self.base_lr * 2, 'weight_decay': 0})
        params.append(
            {'params': normal_bn, 'lr': self.base_lr, 'weight_decay': 0})