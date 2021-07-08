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
        with torch.no_grad():
            param_groups_offload = tuple(torch.nn.Parameter(
                torch.empty_like(p, device=offload_device), requires_grad=p.requires_grad) for p in param_groups)
            for param_main, param_offload in zip(param_groups, param_groups_offload):
                param_offload.copy_(param_main, non_blocking=True)
                if param_offload.grad is None:
                    param_offload.grad = torch.zeros_like(param_offload)
        super().__init__(optim_cls(param_groups_offload, *args, **kwargs))

    @property
    def param_groups(self):
        assert len(self.param_groups_main) == len(self.optim.param_groups)
        merged_param_groups = []
        for main_pg, offload_pg in zip(self.param_groups_main, self.optim.param_groups):
            merged_param_groups.append(dict(offload_pg, **main_pg))  # override parameters
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
                for param in group:
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
