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

# multi-head self attention with tree feature
class TreeFeatureMHSA(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_dim: int,
                 feature_dim: int,
                 layer: int,
                 dropout: float,
                 tree_feature_usage: str,
                 n_head: int,
                 initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TreeFeatureMHSA, self).__init__(vocab, regularizer)

        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._layer = layer
        self._dropout = dropout
        self._tree_feature_usage = tree_feature_usage
        self._n_head = n_head
        self._input_dim_each_head = input_dim//n_head
        self._feature_dim_each_head = feature_dim//n_head

        self.tf_embedding = torch.nn.Embedding(vocab.get_vocab_size('tf_labels'), self._feature_dim)
        torch.nn.init.xavier_normal_(self.tf_embedding.weight, gain=3)

        # self.W = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.softmax_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        for idx in range(self._layer):
            # _W = torch.nn.Linear(self._input_dim, 1, bias=False)
            # torch.nn.init.xavier_normal_(_W.weight)
            self.W = torch.nn.Parameter(torch.FloatTensor(self._n_head, 1, 1, self._input_dim_each_head, 1), requires_grad=True)
            torch.nn.init.xavier_normal_(self.W.data)
            # self.W.append(_W)
            self.dropout_layers.append(torch.nn.Dropout(p=self._dropout))
            self.softmax_layers.append(torch.nn.Softmax(dim=2))
            self.norm_layers.append(torch.nn.LayerNorm(input_dim))

        if self._tree_feature_usage == 'concat':
            _W = torch.nn.Linear(self._input_dim, self._feature_dim, bias=False)
            torch.nn.init.xavier_normal_(_W.weight)
            self._A_network = TimeDistributed(_W)

        # self.tf_embedding = torch.nn.Embedding(vocab.get_vocab_size('tf_labels'), self._feature_dim)
        # torch.nn.init.xavier_normal_(self.tf_embedding.weight, gain=3)
        #
        # self.W_1 = torch.nn.Linear(self._feature_dim, self._input_dim, bias=False)
        # torch.nn.init.xavier_normal_(self.W_1.weight)
        #
        # self.W_2 = torch.nn.Linear(self._feature_dim, self._input_dim, bias=False)
        # torch.nn.init.xavier_normal_(self.W_2.weight)
        #
        # self.W_3 = torch.nn.Parameter(torch.FloatTensor(self._n_head, 1, 1, self._input_dim_each_head, 1), requires_grad=True)
        # torch.nn.init.xavier_normal_(self.W_3.data)
        #
        # self.softmax = torch.nn.Softmax(dim=2)
        #
        # self.W_4 = torch.nn.Linear(self._input_dim, self._feature_dim, bias=False)
        # torch.nn.init.xavier_normal_(self.W_4.weight)
        # # self._A_network = TimeDistributed(self.W_3)
        #
        # self.W_5 = torch.nn.Linear(self._input_dim, 1, bias=False)
        # torch.nn.init.xavier_normal_(self.W_5.weight)
        #
        # self.dropout_layer = torch.nn.Dropout(p=self._dropout)

        # initializer(self)

    def forward(self, tf, text_embeddings, text_mask):
        batch, sequence, _ = text_embeddings.size()
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
            x = x.view(batch, sequence, self._n_head, self._input_dim_each_head)
            x = x.permute(2, 0, 1, 3).contiguous().view(-1, sequence, self._input_dim_each_head) # (n_head*batch, sequence, input_dim/n_head)

            A = A.view(batch, sequence, sequence, self._n_head, self._feature_dim_each_head)
            A = A.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence, self._feature_dim_each_head) # (n_head*batch, sequence, sequence, input_dim/n_head)


            # (batch, sequence, feature_dim, input_dim)
            Ax = torch.matmul(A.transpose(2, 3), x.unsqueeze(1))
            # Ax = Ax / seq_len
            # (batch, sequence, feature_dim, 1)
            # AxW = self.W[l](Ax)
            Ax = Ax.view(self._n_head, batch, sequence, self._feature_dim_each_head, self._input_dim_each_head)
            AxW = torch.matmul(Ax, self.W)  # (n_head, batch, sequence, feature_dim/n_head, 1)
            AxW = AxW.view(self._n_head*batch, sequence, self._feature_dim_each_head, 1)

            # (batch, sequence, feature_dim, 1)
            g = self.softmax_layers[l](AxW)
            # (batch, sequence, input_dim)
            Ax = Ax.view(self._n_head*batch, sequence, self._feature_dim_each_head, self._input_dim_each_head)
            gAx = torch.matmul(g.transpose(2, 3), Ax).squeeze(2)
            gAx = gAx.view(self._n_head, batch, sequence, self._input_dim_each_head)
            gAx = gAx.permute(1, 2, 0, 3).contiguous().view(batch, sequence, -1) # (batch, sequence, input_dim)

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

    # tf: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, input_dim)
    # text_mask: (batch, sequence)
    # def forward(self, tf, text_embeddings, text_mask):
    #
    #     tf_mask = (tf[:, :, :] >= 0).float()
    #     H = text_embeddings
    #
    #     # (batch, sequence, sequence, feature_dim)
    #     F = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1)
    #     Q = F
    #
    #     K = torch.matmul(Q.transpose(2, 3), H.unsqueeze(1))
    #     V = K
    #
    #     A = self.softmax(self.W_5(K))
    #
    #     H_tilde = torch.matmul(A.transpose(2, 3), V).squeeze(2)
    #
    #     H_tilde = self.dropout_layer(H_tilde)
    #
    #     H_tilde = self.W_4(H_tilde) # (batch, sequence, feature_dim)
    #
    #     H_tilde = H_tilde * text_mask.unsqueeze(-1)
    #     return H_tilde

    # multi-head
    # def forward(self, tf, text_embeddings, text_mask):
    #     batch, sequence, _ = text_embeddings.size()
    #     tf_mask = (tf[:, :, :] >= 0).float()
    #     H = text_embeddings
    #     H = H.view(batch, sequence, self._n_head, self._input_dim_each_head)
    #     H = H.permute(2, 0, 1, 3).contiguous().view(-1, sequence, self._input_dim_each_head) # (n_head*batch, sequence, input_dim/n_head)
    #
    #     # (batch, sequence, sequence, feature_dim)
    #     F = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1)
    #     Q = F # (batch, sequence, sequence, feature_dim)
    #     Q = Q.view(batch, sequence, sequence, self._n_head, self._feature_dim_each_head)
    #     Q = Q.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence, self._feature_dim_each_head) # (n_head*batch, sequence, sequence, input_dim/n_head)
    #
    #     K = torch.matmul(Q.transpose(2, 3), H.unsqueeze(1)) # (n_head*batch, sequence, feature_dim/n_head, input_dim/n_head)
    #     V = K
    #
    #     K = K.view(self._n_head, batch, sequence, self._feature_dim_each_head, self._input_dim_each_head)
    #     KW = torch.matmul(K, self.W_3)  # (n_head, batch, sequence, feature_dim/n_head, 1)
    #     KW = KW.view(self._n_head*batch, sequence, self._feature_dim_each_head, 1)
    #     A = self.softmax(KW) # (n_head*batch, sequence, feature_dim/n_head, 1)
    #
    #     H_tilde = torch.matmul(A.transpose(2, 3), V).squeeze(2)
    #     H_tilde = H_tilde.view(self._n_head, batch, sequence, self._input_dim_each_head)
    #     H_tilde = H_tilde.permute(1, 2, 0, 3).contiguous().view(batch, sequence, -1) # (batch, sequence, input_dim)
    #
    #     H_tilde = self.dropout_layer(H_tilde)
    #
    #     H_tilde = self.W_4(H_tilde) # (batch, sequence, feature_dim)
    #
    #     H_tilde = H_tilde * text_mask.unsqueeze(-1)
    #     return H_tilde

    #  Q K V attention multi-head
    # def forward(self, tf, text_embeddings, text_mask):
    #
    #     batch, sequence, _ = text_embeddings.size()
    #     H = text_embeddings # (batch, sequence, input_dim)
    #     H = H.view(batch, sequence, self._n_head, self._input_dim_each_head)
    #     H = H.permute(2, 0, 1, 3).contiguous().view(-1, sequence, self._input_dim_each_head) # (n_head*batch, sequence, input_dim/n_head)
    #
    #     tf_mask = (tf[:, :, :] >= 0).float()
    #     F = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1) # (batch, sequence, sequence, feature_dim)
    #
    #     Q_1 = self.W_1(F) # (batch, sequence, sequence, input_dim)
    #     Q_1 = Q_1.view(batch, sequence, sequence, self._n_head, self._input_dim_each_head)
    #     Q_1 = Q_1.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence, self._input_dim_each_head) # (n_head*batch, sequence, sequence, input_dim/n_head)
    #     K = torch.matmul(Q_1.transpose(2, 3), H.unsqueeze(1)) # (n_head*batch, sequence, input_dim/n_head, input_dim/n_head)
    #
    #     Q_2 = self.W_2(F)  # (batch, sequence, sequence, input_dim)
    #     Q_2 = Q_2.view(batch, sequence, sequence, self._n_head, self._input_dim_each_head)
    #     Q_2 = Q_2.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence,self._input_dim_each_head)  # (n_head*batch, sequence, sequence, input_dim/n_head)
    #     V = torch.matmul(Q_2.transpose(2, 3), H.unsqueeze(1))  # (n_head*batch, sequence, input_dim/n_head, input_dim/n_head)
    #
    #     K = K.view(self._n_head, batch, sequence, self._input_dim_each_head, self._input_dim_each_head)
    #     KW = torch.matmul(K, self.W_3) # (n_head, batch, sequence, input_dim/n_head, 1)
    #     KW = KW.view(self._n_head*batch, sequence, self._input_dim_each_head, 1)
    #     A = self.softmax(KW) # (n_head*batch, sequence, input_dim/n_head, 1)
    #
    #     H_tilde = torch.matmul(A.transpose(2, 3), V).squeeze(2) # (n_head*batch, sequence, input_dim/n_head)
    #     H_tilde = H_tilde.view(self._n_head, batch, sequence, self._input_dim_each_head)
    #     H_tilde = H_tilde.permute(1, 2, 0, 3).contiguous().view(batch, sequence, -1) # (batch, sequence, input_dim)
    #
    #     H_tilde = self.W_4(H_tilde) # (batch, sequence, feature_dim)
    #
    #     H_tilde = H_tilde * text_mask.unsqueeze(-1)
    #     return H_tilde

    # only attention multi-head
    # def forward(self, tf, text_embeddings, text_mask):
    #
    #     batch, sequence, _ = text_embeddings.size()
    #     H = text_embeddings # (batch, sequence, input_dim)
    #     H = H.view(batch, sequence, self._n_head, self._input_dim_each_head)
    #     H = H.permute(2, 0, 1, 3).contiguous().view(-1, sequence, self._input_dim_each_head) # (n_head*batch, sequence, input_dim/n_head)
    #
    #     tf_mask = (tf[:, :, :] >= 0).float()
    #     F = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1) # (batch, sequence, sequence, feature_dim)
    #
    #     Q_1 = F # (batch, sequence, sequence, feature_dim)
    #     Q_1 = Q_1.view(batch, sequence, sequence, self._n_head, self._feature_dim_each_head)
    #     Q_1 = Q_1.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence, self._feature_dim_each_head) # (n_head*batch, sequence, sequence, feature_dim/n_head)
    #     K = torch.matmul(Q_1.transpose(2, 3), H.unsqueeze(1)) # (n_head*batch, sequence, feature_dim/n_head, input_dim/n_head)
    #
    #     Q_2 = F  # (batch, sequence, sequence, feature_dim)
    #     Q_2 = Q_2.view(batch, sequence, sequence, self._n_head, self._feature_dim_each_head)
    #     Q_2 = Q_2.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence,self._feature_dim_each_head)  # (n_head*batch, sequence, sequence, feature_dim/n_head)
    #     V = torch.matmul(Q_2.transpose(2, 3), H.unsqueeze(1))  # (n_head*batch, sequence, feature_dim/n_head, input_dim/n_head)
    #
    #     K = K.view(self._n_head, batch, sequence, self._feature_dim_each_head, self._input_dim_each_head)
    #     KW = torch.matmul(K, self.W_3) # (n_head, batch, sequence, input_dim/n_head, 1)
    #     KW = KW.view(self._n_head*batch, sequence, self._feature_dim_each_head, 1)
    #     A = self.softmax(KW) # (n_head*batch, sequence, feature_dim/n_head, 1)
    #
    #     H_tilde = torch.matmul(A.transpose(2, 3), V).squeeze(2) # (n_head*batch, sequence, input_dim/n_head)
    #     H_tilde = H_tilde.view(self._n_head, batch, sequence, self._input_dim_each_head)
    #     H_tilde = H_tilde.permute(1, 2, 0, 3).contiguous().view(batch, sequence, -1) # (batch, sequence, input_dim)
    #
    #     H_tilde = self.W_4(H_tilde) # (batch, sequence, feature_dim)
    #
    #     H_tilde = H_tilde * text_mask.unsqueeze(-1)
    #     return H_tilde

    # k attention multi-head
    # def forward(self, tf, text_embeddings, text_mask):
    #
    #     batch, sequence, _ = text_embeddings.size()
    #     H = text_embeddings # (batch, sequence, input_dim)
    #     H = H.view(batch, sequence, self._n_head, self._input_dim_each_head)
    #     H = H.permute(2, 0, 1, 3).contiguous().view(-1, sequence, self._input_dim_each_head) # (n_head*batch, sequence, input_dim/n_head)
    #
    #     tf_mask = (tf[:, :, :] >= 0).float()
    #     F = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1) # (batch, sequence, sequence, feature_dim)
    #
    #     Q_1 = self.W_1(F) # (batch, sequence, sequence, feature_dim)
    #     Q_1 = Q_1.view(batch, sequence, sequence, self._n_head, self._input_dim_each_head)
    #     Q_1 = Q_1.permute(3, 0, 1, 2, 4).contiguous().view(-1, sequence, sequence, self._input_dim_each_head) # (n_head*batch, sequence, sequence, feature_dim/n_head)
    #     K = torch.matmul(Q_1.transpose(2, 3), H.unsqueeze(1)) # (n_head*batch, sequence, feature_dim/n_head, input_dim/n_head)
    #
    #     V = K  # (n_head*batch, sequence, feature_dim/n_head, input_dim/n_head)
    #
    #     K = K.view(self._n_head, batch, sequence, self._input_dim_each_head, self._input_dim_each_head)
    #     KW = torch.matmul(K, self.W_3) # (n_head, batch, sequence, input_dim/n_head, 1)
    #     KW = KW.view(self._n_head*batch, sequence, self._input_dim_each_head, 1)
    #     A = self.softmax(KW) # (n_head*batch, sequence, feature_dim/n_head, 1)
    #
    #     H_tilde = torch.matmul(A.transpose(2, 3), V).squeeze(2) # (n_head*batch, sequence, input_dim/n_head)
    #     H_tilde = H_tilde.view(self._n_head, batch, sequence, self._input_dim_each_head)
    #     H_tilde = H_tilde.permute(1, 2, 0, 3).contiguous().view(batch, sequence, -1) # (batch, sequence, input_dim)
    #
    #     H_tilde = self.W_4(H_tilde) # (batch, sequence, feature_dim)
    #
    #     H_tilde = H_tilde * text_mask.unsqueeze(-1)
    #     return H_tilde





