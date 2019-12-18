import logging
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from dygie.training.seq_metrics import SeqMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SeqLabel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 feature_size: int,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_scheme: str = 'flat') -> None:
        super(SeqLabel, self).__init__(vocab, regularizer)

        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size('ner_sequence_labels')

        # TODO(dwadden) think of a better way to enforce this.
        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("O", "ner_sequence_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        self._seq_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                mention_feedforward.get_output_dim(),
                self._n_labels)))

        self._seq_metrics = SeqMetrics(vocab, self._n_labels, null_label, label_scheme)

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

        self._label_scheme = label_scheme

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, Any],
                text_mask: torch.IntTensor,
                token_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                token_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        """
        TODO(dwadden) Write documentation.
        """

        seq_scores = self._seq_scorer(token_embeddings)
        # Give large negative scores to masked-out elements.
        mask = text_mask.unsqueeze(-1)
        seq_scores = util.replace_masked_values(seq_scores, mask, -1e20)
        seq_scores[:, :, 0] *= text_mask

        _, predicted_seq = seq_scores.max(2)

        if self._label_scheme == 'flat':
            pred_spans = self._seq_metrics._decode_flat(predicted_seq, text_mask)
        elif self._label_scheme == 'stacked':
            pred_spans = self._seq_metrics._decode_stacked(predicted_seq, text_mask)
        else:
            raise RuntimeError("invalid label_scheme {}".format(self.label_scheme))

        output_dict = {"predicted_seq": predicted_seq, "predicted_seq_span": pred_spans}

        if token_labels is not None:
            self._seq_metrics(predicted_seq, token_labels, text_mask, self.training)
            seq_scores_flat = seq_scores.view(-1, self._n_labels)
            seq_labels_flat = token_labels.view(-1)
            mask_flat = text_mask.view(-1).bool()

            loss = self._loss(seq_scores_flat[mask_flat], seq_labels_flat[mask_flat])
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):

        raise NotImplementedError

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        seq_precision, seq_recall, seq_f1 = self._seq_metrics.get_metric(reset)
        return {"seq_precision": seq_precision,
                "seq_recall": seq_recall,
                "seq_f1": seq_f1,}
