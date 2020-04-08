import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import itertools
import pickle as pkl

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (Field, ListField, TextField, SpanField, MetadataField, ArrayField, LabelField, IndexField,
                                  SequenceLabelField, AdjacencyField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans


from allennlp.data.fields.span_field import SpanField

from dygie.data.fields.adjacency_field_assym import AdjacencyFieldAssym

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO(dwadden) Add types, unit-test, clean up.

class MissingDict(dict):
    """
    If key isn't there, returns default value. Like defaultdict, but it doesn't store the missing
    keys that were queried.
    """
    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val

def make_cluster_dict(clusters: List[List[List[int]]]) -> Dict[SpanField, int]:
    """
    Returns a dict whose keys are spans, and values are the ID of the cluster of which the span is a
    member.
    """
    return {tuple(span): cluster_id for cluster_id, spans in enumerate(clusters) for span in spans}

def cluster_dict_sentence(cluster_dict: Dict[Tuple[int, int], int], sentence_start: int, sentence_end: int):
    """
    Split cluster dict into clusters in current sentence, and clusters that come later.
    """

    def within_sentence(span):
        return span[0] >= sentence_start and span[1] <= sentence_end

    # Get the within-sentence spans.
    cluster_sent = {span: cluster for span, cluster in cluster_dict.items() if within_sentence(span)}

    ## Create new cluster dict with the within-sentence clusters removed.
    new_cluster_dict = {span: cluster for span, cluster in cluster_dict.items()
                        if span not in cluster_sent}

    return cluster_sent, new_cluster_dict


def format_label_fields(sentence: List[str],
                        ner: List[List[Union[int,str]]],
                        relations: List[List[Union[int,str]]],
                        cluster_tmp: Dict[Tuple[int,int], int],
                        events: List[List[Union[int,str]]],
                        sentence_start: int,
                        tree: Dict[str, Any],
                        tree_match_filter: bool,
                        dep_tree: Dict[str, Any],
                        tf: Dict[str, Any],
                        tree_feature_dict: List[str],
                        use_overlap_rel: bool) -> Tuple[Dict[Tuple[int,int],str],
                                                      Dict[Tuple[Tuple[int,int],Tuple[int,int]],str],
                                                      Dict[Tuple[int,int],int], Dict[Tuple[int, int],str],
                                                    Dict[Tuple[int, int],List[Tuple[int, int]]],
                                                    Dict[str,Any]]:
    """
    Format the label fields, making the following changes:
    1. Span indices should be with respect to sentence, not document.
    2. Return dicts whose keys are spans (or span pairs) and whose values are labels.
    """
    ss = sentence_start
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start-ss, span_end-ss), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    # Relations
    # relation_dict = MissingDict("",
    #     (
    #         ((  (span1_start-ss, span1_end-ss),  (span2_start-ss, span2_end-ss)   ), relation)
    #         for (span1_start, span1_end, span2_start, span2_end, relation) in relations
    #     )
    # )
    relation_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in relations:
        if relation == 'Overlap' and not use_overlap_rel:
            continue
        relation_dict_values.append((((span1_start - ss, span1_end - ss), (span2_start - ss, span2_end - ss)), relation))
    relation_dict = MissingDict("", relation_dict_values)

    # Coref
    cluster_dict = MissingDict(-1,
        (
            (   (span_start-ss, span_end-ss), cluster_id)
            for ((span_start, span_end), cluster_id) in cluster_tmp.items()
        )
    )

    # Events. There are two structures. The `trigger_dict` is a mapping from span pairs to the
    # trigger labels. The `arg_dict` maps from (trigger_span, arg_span) pairs to trigger labels.
    trigger_dict = MissingDict("")
    arg_dict = MissingDict("")
    for event in events:
        the_trigger = event[0]
        the_args = event[1:]
        trigger_dict[the_trigger[0] - ss] = the_trigger[1]
        for the_arg in the_args:
            arg_dict[(the_trigger[0] - ss, (the_arg[0] - ss, the_arg[1] - ss))] = the_arg[2]

    # Syntax
    if 'match' in tree:
        if tree_match_filter and not tree['match']:
            syntax_dict = MissingDict("")
        else:
            syntax_dict = MissingDict("",
                (
                    ((syntax_span[0], syntax_span[1]), syntax)
                    for parent, children, syntax, word, syntax_span in tree['nodes']
                )
            )
    else:
        syntax_dict = MissingDict("")

    # Children in Syntax tree
    if 'match' in tree:
        if tree_match_filter and not tree['match']:
            children_dict = MissingDict([])
        else:
            children_dict = MissingDict([],
                (
                    ((syntax_span[0], syntax_span[1]), [(tree['nodes'][child][4][0],tree['nodes'][child][4][1]) for child in children])
                    for parent, children, syntax, word, syntax_span in tree['nodes']
                )
            )
    else:
        children_dict = MissingDict([])

    # Children in Dep Tree
    # if 'match' in dep_tree:
    #     dep_children_dict = MissingDict([],
    #                                 (
    #                                     ((dep_span[0], dep_span[1]),
    #                                      [(dep_tree['nodes'][child][4][0], dep_tree['nodes'][child][4][1]) for child in
    #                                       children])
    #                                     for parent, children, dep, word, dep_span in dep_tree['nodes']
    #                                 )
    #                                 )
    # else:
    #     dep_children_dict = MissingDict([])

    if 'nodes' in dep_tree:
        dep_children_dict = MissingDict("",
                                    (
                                        ((node_idx, adj_node_idx), "1")
                                        for node_idx, adj_node_idxes in enumerate(dep_tree['nodes']) for adj_node_idx in adj_node_idxes
                                    )
                                    )
    else:
        dep_children_dict = MissingDict("")

    # tf_dict = {}
    # for k in ['F1', 'F2', 'F3', 'F4', 'F5']:
    #     if k in tf:
    #         tf_dict[k] = MissingDict("",
    #                                 (
    #                                     ((i, j), feature)
    #                                     for i, token_i_features in enumerate(tf[k]) for j, feature in enumerate(token_i_features)
    #                                 )
    #                                 )
    #     else:
    #         tf_dict[k] = MissingDict("")

    if len(tree_feature_dict) != 0:
        missingdict_values = []
        for i, _ in enumerate(sentence):
            for j, _ in enumerate(sentence):
                feature = ""
                for k in tree_feature_dict:
                    if k in tf:
                        feature += "#"+tf[k][i][j]
                missingdict_values.append(((i, j), feature))

        tf_dict = MissingDict("", missingdict_values)

    else:
        tf_dict = MissingDict("")

    return ner_dict, relation_dict, cluster_dict, trigger_dict, arg_dict, syntax_dict, children_dict, dep_children_dict, tf_dict

def span_in_tree(span_ix, tree):
    if 'match' in tree:
        # a constituent tree exists
        for node in tree['nodes']:
            if span_ix[0] == node[4][0] and span_ix[1] == node[4][1]:
                return node
        return None
    else:
        return None


@DatasetReader.register("ie_json")
class IEJsonReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 context_width: int = 1,
                 debug: bool = False,
                 lazy: bool = False,
                 label_scheme: str = 'flat',
                 tree_span_filter: bool = False,
                 tree_match_filter: bool = False,
                 tree_feature_dict: List[str] = None,
                 use_overlap_rel: bool = False) -> None:
        super().__init__(lazy)
        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int( (context_width - 1) / 2)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._debug = debug
        # self._n_debug_docs = 10
        # debug feili
        self._n_debug_docs = 2
        self._label_scheme = label_scheme
        self._tree_span_filter = tree_span_filter
        self._tree_match_filter = tree_match_filter
        self._tree_feature_dict = tree_feature_dict
        self.use_overlap_rel = use_overlap_rel

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        # file_path = cached_path(file_path)
        # debug feili
        # file_path = file_path

        with open(file_path, "r") as f:
            lines = f.readlines()
        # If we're debugging, only do the first 10 documents.
        if self._debug:
            # lines = lines[:self._n_debug_docs]
            # lines = [lines[0], lines[1]]
            lines = [lines[230]]

        for line in lines:
            # Loop over the documents.
            sentence_start = 0
            # print("start: {}".format(line))
            js = json.loads(line)
            # print("end")
            doc_key = js["doc_key"]
            dataset = js["dataset"] if "dataset" in js else None

            # If some fields are missing in the data set, fill them with empties.
            # TODO(dwadden) do this more cleanly once things are running.
            n_sentences = len(js["sentences"])
            # TODO(Ulme) Make it so that the
            js["sentence_groups"] = [[self._normalize_word(word) for sentence in js["sentences"][max(0, i-self.k):min(n_sentences, i + self.k + 1)] for word in sentence] for i in range(n_sentences)]
            js["sentence_start_index"] = [sum(len(js["sentences"][i-j-1]) for j in range(min(self.k, i))) if i > 0 else 0 for i in range(n_sentences)]
            js["sentence_end_index"] = [js["sentence_start_index"][i] + len(js["sentences"][i]) for i in range(n_sentences)]
            for sentence_group_nr in range(len(js["sentence_groups"])):
                if len(js["sentence_groups"][sentence_group_nr]) > 300:
                    js["sentence_groups"][sentence_group_nr] = js["sentences"][sentence_group_nr]
                    js["sentence_start_index"][sentence_group_nr] = 0
                    js["sentence_end_index"][sentence_group_nr] = len(js["sentences"][sentence_group_nr])
                    if len(js["sentence_groups"][sentence_group_nr])>300:
                        import ipdb;
            if "clusters" not in js:
                js["clusters"] = []
            for field in ["ner", "relations", "events", 'trees', 'dep', 'tf']:
                if field not in js:
                    js[field] = [[] for _ in range(n_sentences)]

            cluster_dict_doc = make_cluster_dict(js["clusters"])
            #zipped = zip(js["sentences"], js["ner"], js["relations"], js["events"])
            zipped = zip(js["sentences"], js["ner"], js["relations"], js["events"], js["sentence_groups"],
                         js["sentence_start_index"], js["sentence_end_index"], js['trees'], js['dep'], js['tf'])

            # Loop over the sentences.
            for sentence_num, (sentence, ner, relations, events, groups, start_ix, end_ix, tree, dep, tf) in enumerate(zipped):

                sentence_end = sentence_start + len(sentence) - 1
                cluster_tmp, cluster_dict_doc = cluster_dict_sentence(
                    cluster_dict_doc, sentence_start, sentence_end)

                # TODO(dwadden) too many outputs. Re-write as a dictionary.
                # Make span indices relative to sentence instead of document.
                ner_dict, relation_dict, cluster_dict, trigger_dict, argument_dict, syntax_dict, children_dict, dep_children_dict, \
                tf_dict\
                    = format_label_fields(sentence, ner, relations, cluster_tmp, events, sentence_start, tree,
                                          self._tree_match_filter, dep, tf, self._tree_feature_dict, self.use_overlap_rel)
                sentence_start += len(sentence)
                instance = self.text_to_instance(
                    sentence, ner_dict, relation_dict, cluster_dict, trigger_dict, argument_dict,
                    doc_key, dataset, sentence_num, groups, start_ix, end_ix, tree, syntax_dict, children_dict, dep_children_dict, tf_dict)
                yield instance


    @overrides
    def text_to_instance(self,
                         sentence: List[str],
                         ner_dict: Dict[Tuple[int, int], str],
                         relation_dict,
                         cluster_dict,
                         trigger_dict,
                         argument_dict,
                         doc_key: str,
                         dataset: str,
                         sentence_num: int,
                         groups: List[str],
                         start_ix: int,
                         end_ix: int,
                         tree: Dict[str, Any],
                         syntax_dict: Dict[Tuple[int, int], str],
                         children_dict: Dict[Tuple[int, int],List[Tuple[int, int]]],
                         dep_children_dict: Dict[Tuple[int, int],List[Tuple[int, int]]],
                         tf_dict: Dict[Tuple[int, int], Any]):
        """
        TODO(dwadden) document me.
        """

        sentence = [self._normalize_word(word) for word in sentence]

        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        text_field_with_context = TextField([Token(word) for word in groups], self._token_indexers)

        # feili, NER labels. One label per token
        ner_sequence_labels = self._generate_ner_label(sentence, ner_dict)
        ner_sequence_label_field = SequenceLabelField(ner_sequence_labels, text_field, label_namespace="ner_sequence_labels")

        # Put together the metadata.
        metadata = dict(sentence=sentence,
                        ner_dict=ner_dict,
                        relation_dict=relation_dict,
                        cluster_dict=cluster_dict,
                        trigger_dict=trigger_dict,
                        argument_dict=argument_dict,
                        doc_key=doc_key,
                        dataset=dataset,
                        groups=groups,
                        start_ix=start_ix,
                        end_ix=end_ix,
                        sentence_num=sentence_num,
                        seq_dict=ner_sequence_labels,
                        tree=tree,
                        syntax_dict=syntax_dict,
                        children_dict=children_dict,
                        dep_children_dict=dep_children_dict
                        )
        metadata_field = MetadataField(metadata)

        # Trigger labels. One label per token in the input.
        token_trigger_labels = []
        for i in range(len(text_field)):
            token_trigger_labels.append(trigger_dict[i])

        trigger_label_field = SequenceLabelField(token_trigger_labels, text_field,
                                                 label_namespace="trigger_labels")

        # Generate fields for text spans, ner labels, coref labels.
        spans = []
        span_ner_labels = []
        # feili
        span_labels = []
        span_coref_labels = []
        span_syntax_labels = []
        span_children_labels = []
        dep_span_children_labels = []
        # span_children_syntax_labels = []
        span_tree_labels = []
        raw_spans = []
        assert len(syntax_dict) == len(children_dict)
        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            span_ix = (start, end)
            # here we need to consider how to use tree info
            # for example, use_tree, span is in tree, match is true or false
            # if self._tree_span_filter and not self._is_span_in_tree(span_ix, syntax_dict, children_dict):
            #     if len(raw_spans) == 0: # in case that there is no span for this instance
            #         pass
            #     else:
            #         continue
            span_tree_labels.append('1' if self._is_span_in_tree(span_ix, syntax_dict, children_dict) else '')

            span_ner_labels.append(ner_dict[span_ix])
            span_labels.append('' if ner_dict[span_ix] == '' else '1')
            span_coref_labels.append(cluster_dict[span_ix])
            spans.append(SpanField(start, end, text_field))
            span_syntax_labels.append(syntax_dict[span_ix])
            raw_spans.append(span_ix)

            # if len(children_dict[span_ix]) == 0:
            #     children_field = ListField([SpanField(-1, -1, text_field)])
            #     children_syntax_field = SequenceLabelField([''], children_field,
            #                                            label_namespace="span_syntax_labels")
            # else:
            #     children_field = ListField([SpanField(children_span[0], children_span[1], text_field)
            #                for children_span in children_dict[span_ix]])
            #     children_syntax_field = SequenceLabelField([syntax_dict[children_span] for children_span in children_dict[span_ix]],
            #                                                children_field, label_namespace="span_syntax_labels")
            # span_children_labels.append(children_field)
            # span_children_syntax_labels.append(children_syntax_field)

        span_field = ListField(spans)

        for span in raw_spans:

            if len(children_dict[span]) == 0:
                children_field = ListField([IndexField(-1, span_field) ])
            else:
                children_field = []
                for children_span in children_dict[span]:
                    if children_span in raw_spans:
                        children_field.append(IndexField(raw_spans.index(children_span), span_field))
                    else:
                        children_field.append(IndexField(-1, span_field))
                children_field = ListField(children_field)

            span_children_labels.append(children_field)

        # for span in raw_spans:
        #     if len(dep_children_dict[span]) == 0:
        #         children_field = ListField([IndexField(-1, span_field)])
        #     else:
        #         children_field = []
        #         for children_span in dep_children_dict[span]:
        #             if children_span in raw_spans:
        #                 children_field.append(IndexField(raw_spans.index(children_span), span_field))
        #             else:
        #                 children_field.append(IndexField(-1, span_field))
        #         children_field = ListField(children_field)
        #     dep_span_children_labels.append(children_field)

        n_tokens = len(sentence)
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_tokens)]
        dep_adjs = []
        dep_adjs_indices = []
        # tf_indices = {}
        # tf_features = {}
        # for k, v in tf_dict.items():
        #     tf_indices[k] = []
        #     tf_features[k] = []
        tf_indices = []
        tf_features = []
        for token_pair in candidate_indices:
            dep_adj_label = dep_children_dict[token_pair]
            if dep_adj_label:
                dep_adjs_indices.append(token_pair)
                dep_adjs.append(dep_adj_label)

            # for k,v in tf_dict.items():
            #     feature = tf_dict[k][token_pair]
            #     if feature:
            #         tf_indices[k].append(token_pair)
            #         tf_features[k].append(feature)

            feature = tf_dict[token_pair]
            if feature:
                tf_indices.append(token_pair)
                tf_features.append(feature)


        ner_label_field = SequenceLabelField(span_ner_labels, span_field,
                                             label_namespace="ner_labels")
        coref_label_field = SequenceLabelField(span_coref_labels, span_field,
                                               label_namespace="coref_labels")
        # feili
        span_label_field = SequenceLabelField(span_labels, span_field, label_namespace="span_labels")

        # Generate labels for relations and arguments. Only store non-null values.
        # For the arguments, by convention the first span specifies the trigger, and the second
        # specifies the argument. Ideally we'd have an adjacency field between (token, span) pairs
        # for the event arguments field, but AllenNLP doesn't make it possible to express
        # adjacencies between two different sequences.
        n_spans = len(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans)]

        relations = []
        relation_indices = []
        for i, j in candidate_indices:
            span_pair = (span_tuples[i], span_tuples[j])
            relation_label = relation_dict[span_pair]
            if relation_label:
                relation_indices.append((i, j))
                relations.append(relation_label)

        relation_label_field = AdjacencyField(
            indices=relation_indices, sequence_field=span_field, labels=relations,
            label_namespace="relation_labels")

        arguments = []
        argument_indices = []
        n_tokens = len(sentence)
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_spans)]
        for i, j in candidate_indices:
            token_span_pair = (i, span_tuples[j])
            argument_label = argument_dict[token_span_pair]
            if argument_label:
                argument_indices.append((i, j))
                arguments.append(argument_label)

        argument_label_field = AdjacencyFieldAssym(
            indices=argument_indices, row_field=text_field, col_field=span_field, labels=arguments,
            label_namespace="argument_labels")

        # Syntax
        span_syntax_field = SequenceLabelField(span_syntax_labels, span_field, label_namespace="span_syntax_labels")
        span_children_field = ListField(span_children_labels)
        span_tree_field = SequenceLabelField(span_tree_labels, span_field, label_namespace="span_tree_labels")
        # span_children_syntax_field = ListField(span_children_syntax_labels)
        # dep_span_children_field = ListField(dep_span_children_labels)
        dep_span_children_field = AdjacencyField(
            indices=dep_adjs_indices, sequence_field=text_field, labels=dep_adjs,
            label_namespace="dep_adj_labels")

        # tf_f1_field = AdjacencyField(indices=tf_indices['F1'], sequence_field=text_field, labels=tf_features['F1'],
        #     label_namespace="tf_f1_labels")
        # tf_f2_field = AdjacencyField(indices=tf_indices['F2'], sequence_field=text_field, labels=tf_features['F2'],
        #                              label_namespace="tf_f2_labels")
        # tf_f3_field = AdjacencyField(indices=tf_indices['F3'], sequence_field=text_field, labels=tf_features['F3'],
        #                              label_namespace="tf_f3_labels")
        # tf_f4_field = AdjacencyField(indices=tf_indices['F4'], sequence_field=text_field, labels=tf_features['F4'],
        #                              label_namespace="tf_f4_labels")
        # tf_f5_field = AdjacencyField(indices=tf_indices['F5'], sequence_field=text_field, labels=tf_features['F5'],
        #                              label_namespace="tf_f5_labels")

        tf_field = AdjacencyField(indices=tf_indices, sequence_field=text_field, labels=tf_features,
                                     label_namespace="tf_labels")


        # Pull it  all together.
        fields = dict(text=text_field_with_context,
                      spans=span_field,
                      ner_labels=ner_label_field,
                      coref_labels=coref_label_field,
                      trigger_labels=trigger_label_field,
                      argument_labels=argument_label_field,
                      relation_labels=relation_label_field,
                      metadata=metadata_field,
                      span_labels=span_label_field,
                      ner_sequence_labels=ner_sequence_label_field,
                      syntax_labels=span_syntax_field,
                      span_children=span_children_field,
                      span_tree_labels=span_tree_field,
                      dep_span_children=dep_span_children_field,
                      # tf_f1 = tf_f1_field,
                      # tf_f2 = tf_f2_field,
                      # tf_f3 = tf_f3_field,
                      # tf_f4 = tf_f4_field,
                      # tf_f5 = tf_f5_field)
                      tf=tf_field)
                      # span_children_syntax=span_children_syntax_field)

        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        # with open(cache_filename, "rb") as f:
        #     for entry in pkl.load(f):
        #         yield entry
        # debug feili
        pass

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        # with open(cache_filename, "wb") as f:
        #     pkl.dump(instances, f)
        # debug feili
        pass


    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    # feili
    @staticmethod
    def _is_span_in_tree(span, syntax_dict, children_dict):

        if span in syntax_dict:
            return True
        else:
            return False

    def _generate_ner_label(self, sentence: List[str],
                         ner_dict: Dict[Tuple[int, int], str],
                         ):
        # if sentence[0] == 'Employing':
        #     a = 1
        if self._label_scheme == 'flat':
            sentence_label = []
            for token_idx, token in enumerate(sentence):
                label = 'O'
                for span, ner_label, in ner_dict.items():
                    if token_idx == span[0]:
                        label = 'B'
                        break
                    elif token_idx == span[1]:
                        label = 'E'
                        break
                    elif span[0] < token_idx < span[1]:
                        label = 'I'
                        break
                sentence_label.append(label)
        elif self._label_scheme == 'stacked':
            sentence_label = []
            for token_idx, token in enumerate(sentence):
                label = ''
                for span, ner_label, in ner_dict.items():
                    if token_idx == span[0]:
                        label += 'S' if span[0] == span[1] else 'B'
                    elif token_idx == span[1]:
                        label += 'S' if span[0] == span[1] else 'E'
                    elif span[0] < token_idx < span[1]:
                        if label == '':
                            label = 'I'
                        elif label.find("B") == -1 and label.find("E") == -1 and label.find("S") == -1:
                            label = 'I'
                if label == '':
                    label = 'O'
                sentence_label.append(label)
                # if label == 'SE' or label == 'BB' or label=="EE":
                #     a = 1
        else:
            raise RuntimeError("invalid label_scheme {}".format(self._label_scheme))

        return sentence_label

        # for token_idx, token in enumerate(sentence):
        #     label = 'O'
        #     for span, ner_label, in ner_dict.items():
        #         if token_idx == span[0]:
        #             if label != 'O' and label != 'I':
        #                 overlapped_label = 'S' if span[0] == span[1] else 'B'
        #                 logger.info('sentence: {}, token_idx {}, current label "{}", overlapped label "{}"'.format(sentence, token_idx, label, overlapped_label))
        #             else:
        #                 label = 'S' if span[0] == span[1] else 'B'
        #
        #         elif token_idx == span[1]:
        #             if label != 'O' and label != 'I':
        #                 overlapped_label = 'S' if span[0] == span[1] else 'E'
        #                 logger.info('sentence: {}, token_idx {}, current label "{}", overlapped label "{}"'.format(sentence, token_idx, label, overlapped_label))
        #             else:
        #                 label = 'S' if span[0] == span[1] else 'E'
        #         elif span[0] < token_idx < span[1]:
        #             label = 'I'


