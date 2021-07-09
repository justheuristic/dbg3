import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LeanAlbertAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 attention_probs_dropout_prob: float = 0,
                 hidden_dropout_prob: float = 0,
                 layer_norm_eps: float = 1e-12):
        """ Attention layer that does not hog GPU memory """
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size, self.num_attention_heads = hidden_size, num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.dense_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.dense_out = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob, inplace=False)
        self.output_dropout = nn.Dropout(hidden_dropout_prob, inplace=False)

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_, attention_mask=None, output_attentions=False):
        hidden_states_ln = self.layer_norm(hidden_states_)

        qkv_output = self.dense_qkv(hidden_states_ln)
        attention_output, attention_probs = checkpoint(
            self.attention_core, qkv_output, attention_mask)
        projected_context_layer = self.dense_out(attention_output)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = projected_context_layer_dropout + hidden_states_
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

    def attention_core(self, qkv_output, attention_mask):
        query = self.transpose_for_scores(qkv_output[..., :self.hidden_size])
        key = self.transpose_for_scores(qkv_output[..., self.hidden_size: 2 * self.hidden_size])
        value = self.transpose_for_scores(qkv_output[..., 2 * self.hidden_size:])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(2, 1).flatten(2)
        return attention_output, attention_probs
