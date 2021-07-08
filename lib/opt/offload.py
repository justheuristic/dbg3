import math
from typing import Type, Iterable, Dict, Union

import torch
from hivemind import nested_map

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
        with torch.no_grad():
            param_groups_offload = nested_map(lambda p: (
                    torch.nn.Parameter(torch.empty_like(p, device=offload_device), requires_grad=p.requires_grad)
                    if isinstance(p, torch.nn.Parameter) else p), param_groups)

            for group, offload_group in zip(param_groups, param_groups_offload):
                for param, offload_param in zip(group['params'], param_groups_offload['params']):
                    offload_param.copy_(param, non_blocking=True)
                    if offload_param.grad is None:
                        offload_param.grad = torch.zeros_like(offload_param)
        super().__init__(optim_cls(param_groups_offload, *args, **kwargs))

    @property
    def param_groups(self):
        assert len(self.param_groups_main) == len(self.optim.param_groups)
        merged_param_groups = []
        for group, offload_group in zip(self.param_groups_main, self.optim.param_groups):
            merged_param_groups.append(dict(offload_group, **group))  # override parameters
        return merged_param_groups

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support add_param_group.")

    @param_groups.setter
    def param_groups(self, param_groups):
        raise NotImplementedError(f"{self.__class__.__name__} does not support modifying param_groups")

    def step(self, closure=None, *args, **kwargs):
        assert closure is None, "closure not supported in cpu offload mode"
        with torch.no_grad():
            for group, offload_group in zip(self.param_groups_main, self.optim.param_groups):
                for param, offload_param in zip(group["params"], offload_group["params"]):
                    offload_param.copy_(param, non_blocking=True)
                    if param.grad is not None:
                        offload_param.grad.copy_(param.grad, non_blocking=True)

        output = self.optim.step(*args, **kwargs)

        with torch.no_grad():
            for group, offload_group in zip(self.param_groups_main, self.optim.param_groups):
                for param, offload_param in zip(group["params"], offload_group["params"]):
                    param.copy_(offload_param, non_blocking=True)
        return output

    def zero_grad(self, set_to_none: bool = False, *args, **kwargs):
        with torch.no_grad():
            for group in self.param_groups_main:
                for param in group['params']:
                    if param.grad is not None:
                        if set_to_none:
                            param.grad = None
                        else:
                            if param.grad.grad_fn is not None:
                                param.grad.detach_()
                            else:
                                param.grad.requires_grad_(False)
                            param.grad.zero_()
        return super().zero_grad(*args, set_to_none=False, **kwargs)
