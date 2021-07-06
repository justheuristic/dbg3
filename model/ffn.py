import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import get_device_states, set_device_states


class LeanFFN(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory.
    Complete with layer norm, residual and dropout
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation=F.gelu,
                 layer_norm_eps: float = 1e-5,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.dense_i2h = nn.Linear(hidden_size, intermediate_size)
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation
        self.dropout = dropout

    def forward(self, input):
        return self.layer_norm(_LeanFFN.apply(
            input,
            self.dense_i2h.weight, self.dense_i2h.bias,
            self.dense_h2o.weight, self.dense_h2o.bias,
            self.activation, self.dropout, self.training
        ))


class _LeanFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, i2h_weight, i2h_bias, h2o_weight, h20_bias,
                activation, dropout, training):
        ctx._activation, ctx._dropout, ctx._training = activation, dropout, training
        ctx._cpu_rng_state = torch.get_rng_state()
        ctx._device_rng_states = get_device_states(input)

        input_2d = input.view(-1, input.shape[-1])
        hid = F.linear(input_2d, i2h_weight, i2h_bias)
        hid_act = activation(hid)
        out = F.linear(hid_act, h2o_weight, h20_bias)
        out = out.add_(input_2d)
        out = F.dropout(out, dropout, training, inplace=True)
        ctx.save_for_backward(input, hid, i2h_weight, h2o_weight)
        return out.view(*input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_i2h_weight = grad_i2h_bias = grad_h2o_weight = grad_h2o_bias = None
        input, hid, i2h_weight, h2o_weight = ctx.saved_tensors
        torch.set_rng_state(ctx._cpu_rng_state)
        set_device_states(*ctx._device_rng_states)

        input_2d = input.view(-1, input.shape[-1])
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        grad_hid_act = torch.mm(grad_output_2d, h2o_weight)
        with torch.enable_grad():
            hid.requires_grad_(True)
            hid_act = ctx._activation(hid)
            grad_hid, = torch.autograd.grad(hid_act, hid, grad_hid_act)
            hid.requires_grad_(False)

        with torch.no_grad():
            grad_input_2d = torch.mm(grad_hid, i2h_weight) + grad_output_2d
            grad_input = grad_input_2d.view(*grad_output.shape)
            grad_i2h_weight = grad_hid.t().mm(input_2d)
            grad_i2h_bias = grad_hid.sum(0)
            grad_h2o_weight = grad_output_2d.t().mm(hid_act)
            grad_h2o_bias = grad_output_2d.sum(0)
        return grad_input, grad_i2h_weight, grad_i2h_bias, grad_h2o_weight, grad_h2o_bias, None, None, None