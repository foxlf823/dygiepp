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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TreeFeature(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_dim: int,
                 feature_dim: int,
                 layer: int,
                 dropout: float,
                 tree_feature_usage: str,
                 initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TreeFeature, self).__init__(vocab, regularizer)

        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._layer = layer
        self._dropout = dropout
        self._tree_feature_usage = tree_feature_usage

        self.tf_embedding = torch.nn.Embedding(vocab.get_vocab_size('tf_labels'), self._feature_dim)
        torch.nn.init.xavier_normal_(self.tf_embedding.weight, gain=3)

        self.W = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.softmax_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        for idx in range(self._layer):
            _W = torch.nn.Linear(self._input_dim, 1, bias=False)
            torch.nn.init.xavier_normal_(_W.weight)
            self.W.append(_W)
            self.dropout_layers.append(torch.nn.Dropout(p=self._dropout))
            self.softmax_layers.append(torch.nn.Softmax(dim=2))
            self.norm_layers.append(torch.nn.LayerNorm(input_dim))

        if self._tree_feature_usage == 'concat':
            _W = torch.nn.Linear(self._input_dim, self._feature_dim, bias=False)
            torch.nn.init.xavier_normal_(_W.weight)
            self._A_network = TimeDistributed(_W)

        # initializer(self)

    # tf: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, input_dim)
    # text_mask: (batch, sequence)
    def forward(self, tf, text_embeddings, text_mask):

        tf_mask = (tf[:, :, :] >= 0).float()
        # debug
        # b = tf_mask.sum(2).sum(1)
        # for dim1 in b:
        #     if dim1.item() == 0:
        #         pass
        # (batch, sequence, sequence, feature_dim)
        A = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1)
        # (batch, 1, 1, 1)
        seq_len = text_mask.sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        x = text_embeddings

        for l in range(self._layer):
            residual = x
            # (batch, sequence, feature_dim, input_dim)
            Ax = torch.matmul(A.transpose(2, 3), x.unsqueeze(1))
            # Ax = Ax / seq_len
            # (batch, sequence, feature_dim, 1)
            AxW = self.W[l](Ax)
            # (batch, sequence, feature_dim, 1)
            g = self.softmax_layers[l](AxW)
            # (batch, sequence, input_dim)
            gAx = torch.matmul(g.transpose(2, 3), Ax).squeeze(2)
            gAx = self.dropout_layers[l](gAx)
            # (batch, sequence, input_dim)
            if self._tree_feature_usage == 'add':
                x = self.norm_layers[l](gAx + residual)
            else:
                x = gAx

        if self._tree_feature_usage == 'concat':
            x = self._A_network(x)
        x = x * text_mask.unsqueeze(-1)
        return x






