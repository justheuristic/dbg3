import math
from typing import Type, Iterable, Dict, Union

import torch

from .wrapper import OptimizerWrapper


class OffloadOptimizer(OptimizerWrapper):
    r""" A wrapper that stores optimizer statistics and performs updates on the offloaded device (e.g. CPU RAM). """

    def __init__(self, param_groups: Union[Iterable[torch.nn.Parameter], Iterable[Dict]],
                 optim_cls: Type[torch.optim.Optimizer], *args, offload_device=torch.device('cpu'), **kwargs):
        param_groups = list(param_groups)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        self.offload_device = offload_device
        self.param_groups_main = param_groups
        self.params_main = tuple(param for group in param_groups for param in group['params'])
        with torch.no_grad():
            self.params_offload = tuple(torch.nn.Parameter(torch.empty_like(p, device=offload_device),
                                                           requires_grad=p.requires_grad) for p in self.params_main)
            for param_main, param_offload in self.params_offload:
                param_offload.copy_(param_main, non_blocking=True)
                if param_offload.grad is None:
                    param_offload.grad = torch.zeros_like(param_offload)
        super().__init__(optim_cls(self.params_offload, *args, **kwargs))

    @property
    def param_groups(self):
        return self.param_groups_main

    def step(self, closure=None, *args, **kwargs):
        assert closure is None, "closure not supported in cpu offload mode"
        with torch.no_grad():
            for param_main, param_offload in zip(self.params_main, self.params_offload):
                param_offload.copy_(param_main, non_blocking=True)
                if param_main.grad is not None:
                    param_offload.grad.copy_(param_main.grad, non_blocking=True)

        output = self.optim.step(*args, **kwargs)

        with torch.no_grad():
            for param_main, param_offload in zip(self.params_main, self.params_offload):
                param_main.copy_(param_offload, non_blocking=True)
        return output

    def zero_grad(self, set_to_none: bool = False, *args, **kwargs):
        with torch.no_grad():
            for param_main in self.params_main:
                if param_main.grad is not None:
                    if set_to_none:
                        param_main.grad = None
                    else:
                        if param_main.grad.grad_fn is not None:
                            param_main.grad.detach_()
                        else:
                            param_main.grad.requires_grad_(False)
                        param_main.grad.zero_()

        return super().zero_grad(*args, set_to_none=False, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support add_param_group.")