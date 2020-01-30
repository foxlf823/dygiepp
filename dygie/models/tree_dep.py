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


class TreeDep(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 span_emb_dim: int,
                 tree_prop: int = 1,
                 tree_dropout: float = 0.0,
                 tree_children: str = 'attention',
                 initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TreeDep, self).__init__(vocab, regularizer)

        self._span_emb_dim = span_emb_dim
        assert span_emb_dim % 2 == 0

        self.layers = tree_prop

        # self._f_network = torch.nn.ModuleList()
        # for idx in range(self._tree_prop):
        #     tmp = FeedForward(input_dim=span_emb_dim,
        #                               num_layers=1,
        #                               hidden_dims=span_emb_dim,
        #                               activations=torch.nn.ReLU,
        #                               dropout=tree_dropout)
        #     self._f_network.add_module('ff-{}'.format(idx), tmp)

        self.W = torch.nn.ModuleList()
        self.gcn_drop = torch.nn.ModuleList()
        for layer in range(self.layers):
            self.W.append(torch.nn.Linear(span_emb_dim, span_emb_dim))
            self.gcn_drop.append(torch.nn.Dropout(p=tree_dropout))

        initializer(self)

    # adj: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, emb_dim)
    # text_mask: (batch, sequence)
    def forward(self, adj, text_embeddings, text_mask):

        denom = adj.sum(2).unsqueeze(2) + 1
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        gcn_inputs = text_embeddings

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gAxW = self.gcn_drop[l](gAxW)
            gcn_inputs = gAxW * text_mask.unsqueeze(-1)

        return gcn_inputs






