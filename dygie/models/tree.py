import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from dygie.models import shared
from allennlp.modules.span_extractors import EndpointSpanExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Tree(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 feature_size: int,
                 span_emb_dim: int,
                 coref_prop: int = 0,
                 coref_prop_dropout_f: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Tree, self).__init__(vocab, regularizer)

        self._f_network = FeedForward(input_dim=2*span_emb_dim,
                                      num_layers=1,
                                      hidden_dims=span_emb_dim,
                                      activations=torch.nn.Sigmoid(),
                                      dropout=coref_prop_dropout_f)

        # self.antecedent_softmax = torch.nn.Softmax(dim=2)

        self._global_attention = TimeDistributed(torch.nn.Linear(span_emb_dim, 1))

        initializer(self)

    # span_embeddings: (batch, sequence, span_emb_dim)
    # span_children: (batch, sequence, children_num, 1)
    # span_children_mask: (batch, sequence, children_num)
    def forward(self, span_embeddings, span_children, span_children_mask):
        batch, sequence, children_num, _ = span_children.size()
        # (batch, sequence, children_num)
        span_children = span_children.squeeze(-1)

        flat_span_indices = util.flatten_and_batch_shift_indices(span_children, span_embeddings.size(1))
        # (batch, sequence, children_num, span_emb_dim)
        children_span_embeddings = util.batched_index_select(span_embeddings, span_children, flat_span_indices)
        # (batch, sequence, children_num)
        attention_scores = self._global_attention(children_span_embeddings).squeeze(-1)
        # (batch, sequence, children_num)
        attention_scores_softmax = util.masked_softmax(attention_scores, span_children_mask, dim=2)
        # attention_scores_softmax = self.antecedent_softmax(attention_scores)
        # debug feili
        # for dim1 in attention_scores_softmax:
        #     for dim2 in dim1:
        #         pass
        # (batch, sequence, span_emb_dim)
        children_span_embeddings_attentioned = util.weighted_sum(children_span_embeddings, attention_scores_softmax)
        # for dim1 in children_span_embeddings_attentioned:
        #     for dim2 in dim1:
        #         pass
        # (batch, sequence, 2*span_emb_dim)
        f_network_input = torch.cat([span_embeddings, children_span_embeddings_attentioned], dim=-1)
        # (batch, sequence, span_emb_dim)
        f_weights = self._f_network(f_network_input)
        # for dim1 in f_weights:
        #     for dim2 in dim1:
        #         pass
        # (batch, sequence, 1), if f_weights_mask=1, this span has at least one child
        f_weights_mask, _ = span_children_mask.max(dim=-1, keepdim=True)
        # for dim1 in f_weights_mask:
        #     for dim2 in dim1:
        #         pass
        # (batch, sequence, span_emb_dim), let the element of f_weights becomes 1 where f_weights_mask==0
        f_weights = util.replace_masked_values(f_weights, f_weights_mask, 1.0)
        # for dim1 in f_weights:
        #     for dim2 in dim1:
        #         pass
        # (batch, sequence, span_emb_dim)
        # for dim1 in span_embeddings:
        #     for dim2 in dim1:
        #         pass
        combined_span_embeddings = f_weights * span_embeddings + (1.0 - f_weights) * children_span_embeddings_attentioned
        # for dim1 in combined_span_embeddings:
        #     for dim2 in dim1:
        #         pass
        return combined_span_embeddings






