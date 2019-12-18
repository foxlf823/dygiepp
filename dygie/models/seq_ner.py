from os import path
import logging
from typing import Dict, List, Optional
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor, SpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner import NERTagger
from dygie.models.relation import RelationExtractor
from dygie.models.events import EventExtractor
from dygie.training.joint_metrics import JointMetrics

from dygie.models.ner_has_none import NERTagger_Has_None
from dygie.models.seq_label import SeqLabel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("seq_ner")
class SeqNER(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 loss_weights: Dict[str, int],
                 lexical_dropout: float = 0.2,
                 lstm_dropout: float = 0.4,
                 use_attentive_span_extractor: bool = False,
                 co_train: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None,
                 span_extractor: SpanExtractor = None) -> None:
        super(SeqNER, self).__init__(vocab, regularizer)

        # logger.info(vocab.get_index_to_token_vocabulary("ner_sequence_labels"))
        # exit(1)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        # Need to add this line so things don't break. TODO(dwadden) sort out what's happening.
        modules = Params(modules)

        self._ner = NERTagger.from_params(vocab=vocab,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))

        # self._ner = NERTagger_Has_None.from_params(vocab=vocab,
        #                                   feature_size=feature_size,
        #                                   params=modules.pop("ner"))

        self._seq = SeqLabel.from_params(vocab=vocab, feature_size=feature_size, params=modules.pop("seq"))

        self._endpoint_span_extractor = span_extractor

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        if lstm_dropout > 0:
            self._lstm_dropout = torch.nn.Dropout(p=lstm_dropout)
        else:
            self._lstm_dropout = lambda x: x

        initializer(self)


    @overrides
    def forward(self,
                text,
                spans,
                ner_labels,
                coref_labels,
                relation_labels,
                trigger_labels,
                argument_labels,
                metadata,
                span_labels,
                ner_sequence_labels):

        # Shape: (batch_size, max_sentence_length, bert_size)
        text_embeddings = self._text_field_embedder(text)

        text_embeddings = self._lexical_dropout(text_embeddings)

        # Shape: (batch_size, max_sentence_length)
        text_mask = util.get_text_field_mask(text).float()

        sentence_lengths = 0*text_mask.sum(dim=1).long()
        for i in range(len(metadata)):
            sentence_lengths[i] = metadata[i]["end_ix"] - metadata[i]["start_ix"]

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._lstm_dropout(self._context_layer(text_embeddings, text_mask))
        assert spans.max() < contextualized_embeddings.shape[1]

        span_mask = (spans[:, :, 0] >= 0).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans, text_mask)

        # Make calls out to the modules to get results.
        output_ner = {'loss': 0}
        output_seq = {'loss': 0}

        # Make predictions and compute losses for each module
        if self._loss_weights['seq'] > 0:
            output_seq = self._seq(text, text_mask, contextualized_embeddings, sentence_lengths, ner_sequence_labels, metadata)

        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata, output_seq)

        loss = (
                self._loss_weights['ner'] * output_ner['loss'] +
                self._loss_weights['seq'] * output_seq['loss']
                )

        output_dict = dict(ner=output_ner, seq=output_seq)
        output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """
        # TODO(dwadden) which things are already decoded?
        # res = {}
        # if self._loss_weights["ner"] > 0:
        #     res["ner"] = self._ner.decode(output_dict["ner"])
        #
        # return res
        return NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_seq = self._seq.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_ner.keys()) + list(metrics_seq.keys())
                        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
                           list(metrics_ner.items()) + list(metrics_seq.items())
                           )

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res