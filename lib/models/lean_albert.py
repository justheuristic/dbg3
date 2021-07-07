# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ALBERT modules that do not hog your GPU memory """

import torch.nn as nn
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.models.albert import AlbertConfig
from transformers.models.albert.modeling_albert import (
    load_tf_weights_in_albert, AlbertEmbeddings, AlbertMLMHead,
    AlbertSOPHead, ALBERT_START_DOCSTRING, ACT2FN, AlbertLayerGroup, AlbertTransformer, AlbertForPreTraining,
    AlbertModel
)

from transformers.utils import logging
from lib.modules.self_attn import LeanAlbertAttention
from lib.modules.ffn import LeanFFN

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


class LeanAlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attention = LeanAlbertAttention(config.hidden_size, config.num_attention_heads,
                                             attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                             hidden_dropout_prob=config.hidden_dropout_prob,
                                             layer_norm_eps=config.layer_norm_eps)
        self.ffn = LeanFFN(config.hidden_size, config.intermediate_size,
                           activation=ACT2FN[config.hidden_act],
                           layer_norm_eps=config.layer_norm_eps,
                           dropout=config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attention_output, *extras = self.attention(hidden_states, attention_mask, output_attentions)
        ffn_output = self.ffn(attention_output)
        return (ffn_output, attention_output, *extras)


class LeanAlbertLayerGroup(AlbertLayerGroup):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.albert_layers = nn.ModuleList([LeanAlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        if any(head_mask):
            raise NotImplementedError(f"head mask was provided, but it is not supported")

        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class LeanAlbertTransformer(AlbertTransformer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList(
            [LeanAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])


class AlbertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AlbertConfig
    base_model_prefix = "albert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare ALBERT Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class LeanAlbertModel(AlbertModel):
    def __init__(self, config: AlbertConfig, add_pooling_layer=True):
        AlbertPreTrainedModel.__init__(self, config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = LeanAlbertTransformer(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()


class LeanAlbertForPreTraining(AlbertPreTrainedModel, AlbertForPreTraining):
    def __init__(self, config: AlbertConfig):
        AlbertPreTrainedModel.__init__(self, config)

        self.albert = LeanAlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        self.init_weights()
