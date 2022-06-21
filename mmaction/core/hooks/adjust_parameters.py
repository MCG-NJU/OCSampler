from mmcv.runner import Hook
from math import cos, pi


def annealing_cos(start, end, factor, weight=1):
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


class ParamAdjusterHook(Hook):

    def __init__(self, base_ratio=0.5, min_ratio=0., by_epoch=False, style='cos'):
        self.min_ratio = min_ratio
        self.by_epoch = by_epoch
        self.base_ratio = base_ratio
        self.style = style
        assert style in ['cos', 'step']

    def before_epoch(self, runner):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            return

        if self.style == 'cos':
            ratio = annealing_cos(self.base_ratio, self.min_ratio, progress / max_progress)
        elif self.style == 'step':
            ratio = self.base_ratio - (progress / max_progress) * (self.base_ratio - self.min_ratio)
        runner.model.module.explore_rate = ratio

    def before_iter(self, runner):
        if self.by_epoch:
            return
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.style == 'cos':
            ratio = annealing_cos(self.base_ratio, self.min_ratio, progress / max_progress)
        elif self.style == 'step':
            ratio = self.base_ratio - (progress / max_progress) * (self.base_ratio - self.min_ratio)
        runner.model.module.explore_rate = ratio
