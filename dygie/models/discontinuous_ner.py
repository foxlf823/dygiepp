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
from dygie.models.tree import Tree
from dygie.models.tree_dep import TreeDep
from allennlp.modules.token_embedders.embedding import Embedding
from dygie.models.events import EventExtractor
from dygie.training.joint_metrics import JointMetrics
from dygie.models.transformer import MyTransformer
from dygie.models.tree_feature import TreeFeature

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("discontinuous_ner")
class DisNER(Model):
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
                 use_dep: bool,
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
        super(DisNER, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        # Need to add this line so things don't break. TODO(dwadden) sort out what's happening.
        modules = Params(modules)

        self._ner = NERTagger.from_params(vocab=vocab,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))
        self._relation = RelationExtractor.from_params(vocab=vocab,
                                                       feature_size=feature_size,
                                                       params=modules.pop("relation"))

        # Make endpoint span extractor.
        self._endpoint_span_extractor = span_extractor

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        # Do co-training if we're training on ACE and ontonotes.
        self._co_train = co_train

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

        self.use_dep = use_dep
        if self.use_dep:
            # self._dep_tree = Tree.from_params(vocab=vocab,
            #                                   feature_size=feature_size,
            #                                   params=modules.pop("dep_tree"))
            self._dep_tree = TreeDep.from_params(vocab=vocab,
                                              feature_size=feature_size,
                                              params=modules.pop("dep_tree"))

        self.use_tree_feature = use_tree_feature
        if self.use_tree_feature:
            # self._tf_f1_embedding = Embedding(vocab.get_vocab_size('tf_f1_labels'), 1)
            # self._tf_f2_embedding = Embedding(vocab.get_vocab_size('tf_f2_labels'), 1)
            # self._tf_f3_embedding = Embedding(vocab.get_vocab_size('tf_f3_labels'), 1)
            # self._tf_f4_embedding = Embedding(vocab.get_vocab_size('tf_f4_labels'), 1)
            # self._tf_f5_embedding = Embedding(vocab.get_vocab_size('tf_f5_labels'), 1)

            self._tree_feature_arch = tree_feature_arch
            if self._tree_feature_arch == 'transformer':
                self._tf_layer = MyTransformer.from_params(vocab=vocab,
                                                  params=modules.pop("tf_transformer"))
            else:
                self._tf_layer = TreeFeature.from_params(vocab=vocab,
                                                         params=modules.pop("tf_layer"))
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
        """
        TODO(dwadden) change this.
        """

        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        relation_labels = relation_labels.long()

        # debug feili, check relation_labels
        # for dim1 in relation_labels:
        #     for dim2 in dim1:
        #         pass


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

        if self.use_dep:
            dep_span_children = dep_span_children + 1
            contextualized_embeddings = self._dep_tree(dep_span_children, contextualized_embeddings, text_mask)

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
        # if self.use_dep:
        #     dep_span_children_mask = (dep_span_children[:, :, :, 0] >= 0).float()
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
        # if self.use_dep:
        #     dep_span_children = F.relu(dep_span_children.float()).long()

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

        # if self.use_dep:
        #     span_embeddings = self._dep_tree(span_embeddings, dep_span_children, dep_span_children_mask)

        # Make calls out to the modules to get results.
        output_ner = {'loss': 0}
        output_relation = {'loss': 0}


        # Prune and compute span representations for relation module
        if self._loss_weights["relation"] > 0 or self._relation.rel_prop > 0:
            output_relation = self._relation.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        if self._relation.rel_prop > 0:
            output_relation = self._relation.relation_propagation(output_relation)
            span_embeddings = self.update_span_embeddings(span_embeddings, span_mask,
                output_relation["top_span_embeddings"], output_relation["top_span_mask"],
                output_relation["top_span_indices"])

        # Make predictions and compute losses for each module
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)
            self._ner.decode(output_ner)

        if self._loss_weights['relation'] > 0:
            output_relation = self._relation.predict_labels(relation_labels, output_relation, metadata, output_ner)


        if "loss" not in output_relation:
            output_relation["loss"] = 0

        # TODO(dwadden) just did this part.
        loss = (
                self._loss_weights['ner'] * output_ner['loss'] +
                self._loss_weights['relation'] * output_relation['loss']
                )

        output_dict = dict(
                           relation=output_relation,
                           ner=output_ner,
                           )
        output_dict['loss'] = loss

        # Check to see if event predictions are globally compatible (argument labels are compatible
        # with NER tags and trigger tags).
        # if self._loss_weights["ner"] > 0 and self._loss_weights["events"] > 0:
        #     decoded_ner = self._ner.decode(output_dict["ner"])
        #     decoded_events = self._events.decode(output_dict["events"])
        #     self._joint_metrics(decoded_ner, decoded_events)

        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings, top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

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
        # if self._loss_weights["coref"] > 0:
        #     res["coref"] = self._coref.decode(output_dict["coref"])
        # if self._loss_weights["ner"] > 0:
        #     res["ner"] = self._ner.decode(output_dict["ner"])
        # if self._loss_weights["relation"] > 0:
        #     res["relation"] = self._relation.decode(output_dict["relation"])
        # if self._loss_weights["events"] > 0:
        #     res["events"] = output_dict["events"]
        #
        # return res
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """

        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_ner.keys()) +
                        list(metrics_relation.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
                           list(metrics_ner.items()) +
                           list(metrics_relation.items())
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
