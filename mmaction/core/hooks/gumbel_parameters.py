from mmcv.runner import Hook

import numpy as np


def get_current_temperature(num_epoch, init_tau, exp_decay=True, exp_decay_factor=-0.045):
    if exp_decay:
        tau = init_tau * np.exp(exp_decay_factor * num_epoch)
    else:
        tau = init_tau
    return tau


class TauAdjusterHook(Hook):

    def __init__(self, init_tau, exp_decay=True, exp_decay_factor=-0.045):
        self.init_tau = init_tau
        self.exp_decay = exp_decay
        self.exp_decay_factor = exp_decay_factor

    def before_epoch(self, runner):
        tau = get_current_temperature(runner.epoch, self.init_tau, self.exp_decay, self.exp_decay_factor)
        runner.model.module.tau = tau
