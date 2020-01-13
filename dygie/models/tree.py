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
from math import floor
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor import BidirectionalEndpointSpanExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Tree(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 span_emb_dim: int,
                 tree_prop: int = 1,
                 tree_dropout: float = 0.0,
                 tree_children: str = 'attention',
                 initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Tree, self).__init__(vocab, regularizer)

        self._span_emb_dim = span_emb_dim
        assert span_emb_dim % 2 == 0

        self._f_network = FeedForward(input_dim=2*span_emb_dim,
                                      num_layers=1,
                                      hidden_dims=span_emb_dim,
                                      activations=torch.nn.Sigmoid(),
                                      dropout=0)

        self._tree_prop = tree_prop

        self._tree_children = tree_children
        if self._tree_children == 'attention':
            self._global_attention = TimeDistributed(torch.nn.Linear(span_emb_dim, 1))
        elif self._tree_children == 'pooling':
            pass
        elif self._tree_children == 'conv':
            self._conv = torch.nn.Conv1d(span_emb_dim, span_emb_dim, kernel_size=3, padding=1)
        elif self._tree_children == 'rnn':
            self._encoder = PytorchSeq2SeqWrapper(StackedBidirectionalLstm(span_emb_dim, int(floor(span_emb_dim / 2)), 1))
        else:
            raise RuntimeError('invalid tree_children option: {}'.format(self._tree_children))

        self._dropout = torch.nn.Dropout(p=tree_dropout)

        initializer(self)

    # span_embeddings: (batch, sequence, span_emb_dim)
    # span_children: (batch, sequence, children_num, 1)
    # span_children_mask: (batch, sequence, children_num)
    def forward(self, span_embeddings, span_children, span_children_mask):
        batch, sequence, children_num, _ = span_children.size()
        # (batch, sequence, children_num)
        span_children = span_children.squeeze(-1)

        for t in range(self._tree_prop):

            flat_span_indices = util.flatten_and_batch_shift_indices(span_children, span_embeddings.size(1))
            # (batch, sequence, children_num, span_emb_dim)
            children_span_embeddings = util.batched_index_select(span_embeddings, span_children, flat_span_indices)

            if self._tree_children == 'attention':
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
                children_span_embeddings_merged = util.weighted_sum(children_span_embeddings, attention_scores_softmax)
            elif self._tree_children == 'pooling':
                children_span_embeddings_merged = util.masked_max(children_span_embeddings, span_children_mask.unsqueeze(-1), dim=2)
            elif self._tree_children == 'conv':
                masked_children_span_embeddings = children_span_embeddings * span_children_mask.unsqueeze(-1)

                masked_children_span_embeddings = masked_children_span_embeddings.view(batch * sequence, children_num, -1).transpose(1, 2)

                conv_children_span_embeddings = torch.nn.functional.relu(self._conv(masked_children_span_embeddings))

                conv_children_span_embeddings = conv_children_span_embeddings.transpose(1, 2).view(batch, sequence, children_num, -1)

                children_span_embeddings_merged = util.masked_max(conv_children_span_embeddings, span_children_mask.unsqueeze(-1), dim=2)
            elif self._tree_children == 'rnn':
                masked_children_span_embeddings = children_span_embeddings * span_children_mask.unsqueeze(-1)
                masked_children_span_embeddings = masked_children_span_embeddings.view(batch * sequence, children_num, -1)
                try : # if all spans don't have children in this batch, this code will report error
                    rnn_children_span_embeddings = self._encoder(masked_children_span_embeddings, span_children_mask.view(batch * sequence, children_num))
                except Exception as e:
                    rnn_children_span_embeddings = masked_children_span_embeddings

                rnn_children_span_embeddings = rnn_children_span_embeddings.view(batch, sequence, children_num, -1)
                forward_sequence, backward_sequence = rnn_children_span_embeddings.split(int(self._span_emb_dim / 2), dim=-1)
                children_span_embeddings_merged = torch.cat([forward_sequence[:,:,-1,:], backward_sequence[:,:,0,:]], dim=-1)
            else:
                raise RuntimeError
            # for dim1 in children_span_embeddings_attentioned:
            #     for dim2 in dim1:
            #         pass
            # (batch, sequence, 2*span_emb_dim)
            f_network_input = torch.cat([span_embeddings, children_span_embeddings_merged], dim=-1)
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
            span_embeddings = f_weights * span_embeddings + (1.0 - f_weights) * children_span_embeddings_merged
            # for dim1 in combined_span_embeddings:
            #     for dim2 in dim1:
            #         pass

        span_embeddings = self._dropout(span_embeddings)

        return span_embeddings






