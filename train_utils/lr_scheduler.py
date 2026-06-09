import types
import math

from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from train_utils import resnet20


class CooldownLR(_LRScheduler):
    """
    Learning rate cooldown to 0.05× the initial learning rate over the last 500 steps of training
    https://arxiv.org/abs/2306.08153
    """

    def __init__(self, optimizer,
                 factor=1.0 / 3,
                 total_steps=5,
                 cooldown_steps=500,
                 steps_one_epoch=100,
                 last_epoch=-1,
                 verbose="deprecated"):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_steps = total_steps
        self.cooldown_steps = cooldown_steps
        self.steps_one_epoch = steps_one_epoch
        self.cooldown_epoch = (total_steps - cooldown_steps) // steps_one_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch < self.cooldown_epoch:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * self.factor for group in self.optimizer.param_groups]

    def get_lr_list(self):
        num_epoch = self.total_steps // self.steps_one_epoch
        lr_list = []
        for epoch in range(num_epoch):
            if epoch < self.cooldown_epoch:
                lr_list.extend([group['lr'] for group in self.optimizer.param_groups])
            else:
                lr_list.extend([group['lr'] * self.factor for group in self.optimizer.param_groups])
        return lr_list


class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self,
                 optimizer,
                 T_max,
                 num_epoch,
                 eta_min=0,
                 last_epoch=-1,
                 verbose="deprecated"):
        self.T_max = T_max
        self.eta_min = eta_min
        self.num_epoch = num_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

    def get_lr_list(self):
        lr_list = []
        for epoch in range(self.num_epoch):
            lr_list.extend([self.eta_min + (base_lr - self.eta_min) *
                            (1 + math.cos(math.pi * epoch / self.T_max)) / 2
                            for base_lr in self.base_lrs])
        return lr_list


if __name__ == '__main__':
    model = resnet20(num_classes=10)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.5,
                                momentum=0.,
                                weight_decay=0.)
    scheduler = CooldownLR(optimizer, steps=500)
    print(scheduler)
