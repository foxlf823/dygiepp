
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from typing import Any, Dict, List, Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask, tf_f1_features, tf_f2_features, tf_f3_features,
                tf_f4_features, tf_f5_features):

        attn = torch.bmm(q, k.transpose(1, 2)) + tf_f1_features + tf_f2_features + tf_f3_features + tf_f4_features + \
                                                tf_f5_features
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, vocab, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self._tf_f1_embedding = nn.Embedding(vocab.get_vocab_size('tf_f1_labels'), n_head)
        self._tf_f2_embedding = nn.Embedding(vocab.get_vocab_size('tf_f2_labels'), n_head)
        self._tf_f3_embedding = nn.Embedding(vocab.get_vocab_size('tf_f3_labels'), n_head)
        self._tf_f4_embedding = nn.Embedding(vocab.get_vocab_size('tf_f4_labels'), n_head)
        self._tf_f5_embedding = nn.Embedding(vocab.get_vocab_size('tf_f5_labels'), n_head)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    # def forward(self, q, k, v, mask, tf_f1_features, tf_f2_features, tf_f3_features,
    #             tf_f4_features, tf_f5_features, tf_mask):
    def forward(self, q, k, v, mask, tf_f1, tf_f2, tf_f3,
                tf_f4, tf_f5):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        tf_mask = (tf_f1[:, :, :] >= 0).float()
        tf_f1_features = self._tf_f1_embedding((tf_f1 * tf_mask).long()) * tf_mask.unsqueeze(-1)
        tf_f2_features = self._tf_f2_embedding((tf_f2 * tf_mask).long()) * tf_mask.unsqueeze(-1)
        tf_f3_features = self._tf_f3_embedding((tf_f3 * tf_mask).long()) * tf_mask.unsqueeze(-1)
        tf_f4_features = self._tf_f4_embedding((tf_f4 * tf_mask).long()) * tf_mask.unsqueeze(-1)
        tf_f5_features = self._tf_f5_embedding((tf_f5 * tf_mask).long()) * tf_mask.unsqueeze(-1)

        # tf_mask = tf_mask.repeat(n_head, 1, 1)
        tf_f1_features = tf_f1_features.permute(3, 0, 1, 2).contiguous().view(-1, len_q, len_q)
        tf_f2_features = tf_f2_features.permute(3, 0, 1, 2).contiguous().view(-1, len_q, len_q)
        tf_f3_features = tf_f3_features.permute(3, 0, 1, 2).contiguous().view(-1, len_q, len_q)
        tf_f4_features = tf_f4_features.permute(3, 0, 1, 2).contiguous().view(-1, len_q, len_q)
        tf_f5_features = tf_f5_features.permute(3, 0, 1, 2).contiguous().view(-1, len_q, len_q)

        output, attn = self.attention(q, k, v, mask, tf_f1_features, tf_f2_features, tf_f3_features,
                tf_f4_features, tf_f5_features)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, vocab, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(vocab, n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    # def forward(self, enc_input, non_pad_mask, slf_attn_mask, tf_f1_features, tf_f2_features, tf_f3_features,
    #             tf_f4_features, tf_f5_features, tf_mask):
    def forward(self, enc_input, non_pad_mask, slf_attn_mask, tf_f1, tf_f2, tf_f3,
                tf_f4, tf_f5):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, slf_attn_mask, tf_f1, tf_f2, tf_f3,
                tf_f4, tf_f5)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class MyTransformer(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 d_input,
                 d_inner,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 dropout,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MyTransformer, self).__init__(vocab, regularizer)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(vocab, d_input, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    # def forward(self, contextualized_embeddings, text_mask, tf_f1_features, tf_f2_features, tf_f3_features,
    #                              tf_f4_features, tf_f5_features, tf_mask):
    def forward(self, contextualized_embeddings, text_mask, tf_f1, tf_f2, tf_f3, tf_f4, tf_f5):

        # -- Prepare masks
        slf_attn_mask = text_mask == 0
        slf_attn_mask = slf_attn_mask.unsqueeze(1).expand(-1, text_mask.size(1), -1)

        non_pad_mask = text_mask.unsqueeze(-1)

        enc_output = contextualized_embeddings

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask,
                slf_attn_mask,
                tf_f1, tf_f2, tf_f3,
                tf_f4, tf_f5)

        return enc_output