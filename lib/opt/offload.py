import contextlib
import math
from typing import Type, Iterable, Dict, Union

import torch
from hivemind import nested_map

from .wrapper import OptimizerWrapper


class OffloadOptimizer(OptimizerWrapper):
    r""" A wrapper that stores optimizer statistics and performs updates on the offloaded device (e.g. CPU RAM). """

    def __init__(
            self, param_groups: Union[Iterable[torch.nn.Parameter], Iterable[Dict]],
            optim_cls: Type[torch.optim.Optimizer], *args, full_sync: bool = True,
            offload_device=torch.device('cpu'), **kwargs):
        param_groups = list(param_groups)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        with torch.no_grad():
            self.offload_params_per_group = tuple([
                torch.nn.Parameter(torch.empty_like(param, device=offload_device), requires_grad=param.requires_grad)
                for param in group["params"]] for group in param_groups)

            for group, offload_params in zip(param_groups, self.offload_params_per_group):
                for param, offload_param in zip(group['params'], offload_params):
                    offload_param.copy_(param, non_blocking=True)
                    if offload_param.grad is None:
                        offload_param.grad = torch.zeros_like(offload_param)
                    if param.grad is not None:
                        offload_param.grad.copy_(param.grad, non_blocking=True)

        self.full_sync = full_sync
        with self._replace_params(param_groups,
                                  sync_params_before=False, sync_grads_before=False,
                                  sync_params_after=True, sync_grads_after=full_sync):
            super().__init__(optim_cls(param_groups, *args, **kwargs))

    @contextlib.contextmanager
    def _replace_params(self, param_groups, *,
                        sync_params_before: bool, sync_grads_before: bool,
                        sync_params_after: bool, sync_grads_after:bool):
        assert len(param_groups) == len(self.offload_params_per_group)
        normal_params_per_group = [group["params"] for group in param_groups]
        try:
            with torch.no_grad():
                for normal_params, offload_params in zip(normal_params_per_group, self.offload_params_per_group):
                    for param, offload_param in zip(normal_params, offload_params):
                        if sync_params_before:
                            offload_param.copy_(param, non_blocking=True)
                        if sync_grads_before and param.grad is not None:
                            offload_param.grad.copy_(param.grad, non_blocking=True)

            for group, offload_params in zip(param_groups, self.offload_params_per_group):
                group["params"] = offload_params
            yield param_groups
        finally:
            for group, normal_params in zip(param_groups, normal_params_per_group):
                group["params"] = normal_params

            with torch.no_grad():
                for normal_params, offload_params in zip(normal_params_per_group, self.offload_params_per_group):
                    for param, offload_param in zip(normal_params, offload_params):
                        if sync_params_after:
                            param.copy_(offload_param, non_blocking=True)
                        if sync_grads_after and param.grad is not None:
                            param.grad.copy_(offload_param.grad)

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support add_param_group.")

    def step(self, closure=None, *args, **kwargs):
        assert closure is None, "closure not supported in cpu offload mode"
        with self._replace_params(self.param_groups,
                                  sync_params_before=self.full_sync, sync_grads_before=True,
                                  sync_params_after=True, sync_grads_after=self.full_sync):
            return self.optim.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False, *args, **kwargs):
        if not self.full_sync:
            torch.optim.Optimizer.zero_grad(self, set_to_none)
        with self._replace_params(self.param_groups,
                                  sync_params_before=self.full_sync, sync_grads_before=self.full_sync,
                                  sync_params_after=self.full_sync, sync_grads_after=self.full_sync):
            return super().zero_grad(*args, set_to_none=False, **kwargs)

    def state_dict(self):
        with self._replace_params(self.param_groups,
                                  sync_params_before=self.full_sync, sync_grads_before=self.full_sync,
                                  sync_params_after=False, sync_grads_after=False):
            return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        with self._replace_params(self.param_groups,
                                  sync_params_before=False, sync_grads_before=False,
                                  sync_params_after=True, sync_grads_after=self.full_sync):
            return self.optim.load_state_dict(state_dict)
