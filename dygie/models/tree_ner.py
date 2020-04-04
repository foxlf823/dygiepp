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
from dygie.models.tree import Tree

from allennlp.modules.token_embedders.embedding import Embedding
from dygie.models.transformer import MyTransformer
from dygie.models.tree_feature import TreeFeature
from dygie.models.tree_feature_mhsa import TreeFeatureMHSA

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("tree_ner")
class TreeNer(Model):
    """
    TODO(dwadden) document me.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    submodule_params: ``TODO(dwadden)``
        A nested dictionary specifying parameters to be passed on to initialize submodules.
    max_span_width: ``int``
        The maximum width of candidate spans.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    display_metrics: ``List[str]``. A list of the metrics that should be printed out during model
        training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 loss_weights: Dict[str, int],
                 use_tree: bool,
                 use_syntax: bool,
                 use_tree_feature: bool,
                 tree_feature_first: bool,
                 tree_feature_usage: str,
                 tree_feature_arch: str,
                 tree_span_filter: bool = False,
                 lexical_dropout: float = 0.2,
                 lstm_dropout: float = 0.4,
                 use_attentive_span_extractor: bool = False,
                 co_train: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None,
                 span_extractor: SpanExtractor = None) -> None:
        super(TreeNer, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        # Need to add this line so things don't break. TODO(dwadden) sort out what's happening.
        modules = Params(modules)
        self._coref = CorefResolver.from_params(vocab=vocab,
                                                feature_size=feature_size,
                                                params=modules.pop("coref"))
        self._ner = NERTagger.from_params(vocab=vocab,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))

        self._endpoint_span_extractor = span_extractor

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        # Big gotcha: PyTorch doesn't add dropout to the LSTM's output layer. We need to do this
        # manually.
        if lstm_dropout > 0:
            self._lstm_dropout = torch.nn.Dropout(p=lstm_dropout)
        else:
            self._lstm_dropout = lambda x: x

        self.use_tree = use_tree
        if self.use_tree:
            self.tree_feature_first = tree_feature_first
            self.use_syntax = use_syntax
            if self.use_syntax:
                self._syntax_embedding = Embedding(vocab.get_vocab_size('span_syntax_labels'), feature_size)
            self._tree = Tree.from_params(vocab=vocab,
                                              feature_size=feature_size,
                                              params=modules.pop("tree"))

            self.tree_span_filter = tree_span_filter
            if self.tree_span_filter:
                self._tree_span_embedding = Embedding(vocab.get_vocab_size('span_tree_labels'), feature_size)

        self.use_tree_feature = use_tree_feature
        if self.use_tree_feature:
            # self._tf_f1_embedding = Embedding(vocab.get_vocab_size('tf_f1_labels'), 1)
            # self._tf_f2_embedding = Embedding(vocab.get_vocab_size('tf_f2_labels'), 1)
            # self._tf_f3_embedding = Embedding(vocab.get_vocab_size('tf_f3_labels'), 1)
            # self._tf_f4_embedding = Embedding(vocab.get_vocab_size('tf_f4_labels'), 1)
            # self._tf_f5_embedding = Embedding(vocab.get_vocab_size('tf_f5_labels'), 1)
            # self._tf_transformer = MyTransformer.from_params(vocab=vocab,
            #                                   params=modules.pop("tf_transformer"))

            self._tree_feature_arch = tree_feature_arch
            if self._tree_feature_arch == 'transformer':
                self._tf_layer = MyTransformer.from_params(vocab=vocab,
                                                  params=modules.pop("tf_transformer"))
            elif self._tree_feature_arch == 'gcn':
                self._tf_layer = TreeFeature.from_params(vocab=vocab,
                                                         params=modules.pop("tf_layer"))
            elif self._tree_feature_arch == 'mhsa':
                self._tf_layer = TreeFeatureMHSA.from_params(vocab=vocab,
                                                         params=modules.pop("tf_mhsa"))
            else:
                raise RuntimeError("wrong tree_feature_arch: {}".format(self._tree_feature_arch))
            self._tree_feature_usage = tree_feature_usage

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
                ner_sequence_labels,
                syntax_labels,
                span_children,
                span_tree_labels,
                dep_span_children,
                # tf_f1, tf_f2, tf_f3, tf_f4, tf_f5,
                tf,
                # span_children_syntax
                ):

        text_embeddings = self._text_field_embedder(text)

        # debug feili, check span_children
        # for dim1 in span_children:
        #     for dim2 in dim1:
        #         pass

        text_embeddings = self._lexical_dropout(text_embeddings)

        # Shape: (batch_size, max_sentence_length)
        text_mask = util.get_text_field_mask(text).float()

        sentence_lengths = 0*text_mask.sum(dim=1).long()
        for i in range(len(metadata)):
            sentence_lengths[i] = metadata[i]["end_ix"] - metadata[i]["start_ix"]

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._lstm_dropout(self._context_layer(text_embeddings, text_mask))
        assert spans.max() < contextualized_embeddings.shape[1]

        if self.use_tree_feature:
            # tf_mask = (tf_f1[:, :, :] >= 0).float()
            # tf_f1_features = self._tf_f1_embedding((tf_f1*tf_mask).long()) * tf_mask.unsqueeze(-1)
            # tf_f2_features = self._tf_f2_embedding((tf_f2*tf_mask).long()) * tf_mask.unsqueeze(-1)
            # tf_f3_features = self._tf_f3_embedding((tf_f3*tf_mask).long()) * tf_mask.unsqueeze(-1)
            # tf_f4_features = self._tf_f4_embedding((tf_f4*tf_mask).long()) * tf_mask.unsqueeze(-1)
            # tf_f5_features = self._tf_f5_embedding((tf_f5*tf_mask).long()) * tf_mask.unsqueeze(-1)
            # self._tf_transformer(contextualized_embeddings, text_mask, tf_f1_features, tf_f2_features, tf_f3_features,
            #                      tf_f4_features, tf_f5_features, tf_mask)

            # contextualized_embeddings = self._tf_transformer(contextualized_embeddings, text_mask, tf_f1, tf_f2, tf_f3,
            #                      tf_f4, tf_f5)

            if self._tree_feature_usage == 'add':
                contextualized_embeddings = self._tf_layer(tf, contextualized_embeddings, text_mask)
            else:
                tree_feature_embeddings = self._tf_layer(tf, contextualized_embeddings, text_mask)
                contextualized_embeddings = torch.cat([contextualized_embeddings, tree_feature_embeddings], dim=-1)

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).float()
        if self.use_tree:
            span_children_mask = (span_children[:, :, :, 0] >= 0).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()
        if self.use_tree:
            span_children = F.relu(span_children.float()).long()

        # debug feili, check span_children
        # for dim1, dim1_ in zip(span_children, span_children_mask):
        #     for dim2, dim2_ in zip(dim1, dim1_):
        #         pass

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans, text_mask)

        if self.use_tree:
            if self.tree_feature_first:
                span_feature_group = [span_embeddings]
                if self.use_syntax:
                    syntax_embeddings = self._syntax_embedding(syntax_labels)
                    span_feature_group.append(syntax_embeddings)
                if self.tree_span_filter:
                    tree_span_embeddings = self._tree_span_embedding(span_tree_labels)
                    span_feature_group.append(tree_span_embeddings)

                span_embeddings = torch.cat(span_feature_group, -1)
                span_embeddings = self._tree(span_embeddings, span_children, span_children_mask)
            else:
                span_embeddings = self._tree(span_embeddings, span_children, span_children_mask)

                span_feature_group = [span_embeddings]
                if self.use_syntax:
                    syntax_embeddings = self._syntax_embedding(syntax_labels)
                    span_feature_group.append(syntax_embeddings)
                if self.tree_span_filter:
                    tree_span_embeddings = self._tree_span_embedding(span_tree_labels)
                    span_feature_group.append(tree_span_embeddings)

                span_embeddings = torch.cat(span_feature_group, -1)



        # Make calls out to the modules to get results.
        output_coref = {'loss': 0}
        output_ner = {'loss': 0}

        # Prune and compute span representations for coreference module
        if self._loss_weights["coref"] > 0 or self._coref.coref_prop > 0:
            output_coref, coref_indices = self._coref.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, coref_labels, metadata)

        # Propagation of global information to enhance the span embeddings
        if self._coref.coref_prop > 0:
            # TODO(Ulme) Implement Coref Propagation
            output_coref = self._coref.coref_propagation(output_coref)
            span_embeddings = self._coref.update_spans(output_coref, span_embeddings, coref_indices)

        # Make predictions and compute losses for each module
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)
            self._ner.decode(output_ner)

        if self._loss_weights['coref'] > 0:
            output_coref = self._coref.predict_labels(output_coref, metadata)

        if "loss" not in output_coref:
            output_coref["loss"] = 0

        # TODO(dwadden) just did this part.
        loss = (self._loss_weights['coref'] * output_coref['loss'] +
                self._loss_weights['ner'] * output_ner['loss'] )

        output_dict = dict(coref=output_coref,
                           ner=output_ner)
        output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):


        res = {}
        res["ner"] = output_dict['ner']['decoded_ner']
        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_ner = self._ner.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_coref.keys()) + list(metrics_ner.keys())
                        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_coref.items()) +
                           list(metrics_ner.items())
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
