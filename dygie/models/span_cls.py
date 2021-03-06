import logging
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from dygie.training.span_metrics import SpanMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanCls(Model):
    """
    Named entity recognition module of DyGIE model.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 feature_size: int,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SpanCls, self).__init__(vocab, regularizer)

        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size('span_labels')

        # TODO(dwadden) think of a better way to enforce this.
        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("", "span_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        self._span_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                mention_feedforward.get_output_dim(),
                self._n_labels)))

        self._span_metrics = SpanMetrics(self._n_labels, null_label)

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                span_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        """
        TODO(dwadden) Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings
        span_scores = self._span_scorer(span_embeddings)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        span_scores = util.replace_masked_values(span_scores, mask, -1e20)
        span_scores[:, :, 0] *= span_mask

        _, predicted_span = span_scores.max(2)

        output_dict = {"spans": spans,
                       "span_mask": span_mask,
                       "span_scores": span_scores,
                       "predicted_span": predicted_span}

        if span_labels is not None:
            self._span_metrics(predicted_span, span_labels, span_mask)
            span_scores_flat = span_scores.view(-1, self._n_labels)
            span_labels_flat = span_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(span_scores_flat[mask_flat], span_labels_flat[mask_flat])
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        # predicted_ner_batch = output_dict["predicted_ner"].detach().cpu()
        # spans_batch = output_dict["spans"].detach().cpu()
        # span_mask_batch = output_dict["span_mask"].detach().cpu().bool()
        #
        # res_list = []
        # res_dict = []
        # for spans, span_mask, predicted_NERs in zip(spans_batch, span_mask_batch, predicted_ner_batch):
        #     entry_list = []
        #     entry_dict = {}
        #     for span, ner in zip(spans[span_mask], predicted_NERs[span_mask]):
        #         ner = ner.item()
        #         if ner > 0:
        #             the_span = (span[0].item(), span[1].item())
        #             the_label = self.vocab.get_token_from_index(ner, "ner_labels")
        #             entry_list.append((the_span[0], the_span[1], the_label))
        #             entry_dict[the_span] = the_label
        #     res_list.append(entry_list)
        #     res_dict.append(entry_dict)
        #
        # output_dict["decoded_ner"] = res_list
        # output_dict["decoded_ner_dict"] = res_dict
        # return output_dict

        raise NotImplementedError

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        span_precision, span_recall, span_f1, span_accuracy = self._span_metrics.get_metric(reset)
        return {"span_precision": span_precision,
                "span_recall": span_recall,
                "span_f1": span_f1,
                "span_accuracy": span_accuracy}
