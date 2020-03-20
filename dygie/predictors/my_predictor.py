
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from allennlp.common.util import JsonDict
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
                        tree_feature_dict: List[str]) -> Tuple[Dict[Tuple[int,int],str],
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
    relation_dict = MissingDict("",
        (
            ((  (span1_start-ss, span1_end-ss),  (span2_start-ss, span2_end-ss)   ), relation)
            for (span1_start, span1_end, span2_start, span2_end, relation) in relations
        )
    )

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


@Predictor.register('paper-classifier')
class PaperClassifierPredictor(Predictor):
    """Predictor wrapper for the AcademicPaperClassifier"""
    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        js = json_dict
        sentence_start = 0
        doc_key = js["doc_key"]
        dataset = js["dataset"] if "dataset" in js else None

        # If some fields are missing in the data set, fill them with empties.
        # TODO(dwadden) do this more cleanly once things are running.
        n_sentences = len(js["sentences"])
        # TODO(Ulme) Make it so that the
        js["sentence_groups"] = [[word for sentence in
                                  js["sentences"][max(0, i):min(n_sentences, i + 1)] for word in
                                  sentence] for i in range(n_sentences)]
        js["sentence_start_index"] = [
            sum(len(js["sentences"][i - j - 1]) for j in range(min(0, i))) if i > 0 else 0 for i in
            range(n_sentences)]
        js["sentence_end_index"] = [js["sentence_start_index"][i] + len(js["sentences"][i]) for i in range(n_sentences)]
        for sentence_group_nr in range(len(js["sentence_groups"])):
            if len(js["sentence_groups"][sentence_group_nr]) > 300:
                js["sentence_groups"][sentence_group_nr] = js["sentences"][sentence_group_nr]
                js["sentence_start_index"][sentence_group_nr] = 0
                js["sentence_end_index"][sentence_group_nr] = len(js["sentences"][sentence_group_nr])
                if len(js["sentence_groups"][sentence_group_nr]) > 300:
                    import ipdb;
        if "clusters" not in js:
            js["clusters"] = []
        for field in ["ner", "relations", "events", 'trees', 'dep', 'tf']:
            if field not in js:
                js[field] = [[] for _ in range(n_sentences)]

        cluster_dict_doc = make_cluster_dict(js["clusters"])
        # zipped = zip(js["sentences"], js["ner"], js["relations"], js["events"])
        zipped = zip(js["sentences"], js["ner"], js["relations"], js["events"], js["sentence_groups"],
                     js["sentence_start_index"], js["sentence_end_index"], js['trees'], js['dep'], js['tf'])

        outputs = []
        # Loop over the sentences.
        for sentence_num, (sentence, ner, relations, events, groups, start_ix, end_ix, tree, dep, tf) in enumerate(
                zipped):
            sentence_end = sentence_start + len(sentence) - 1
            cluster_tmp, cluster_dict_doc = cluster_dict_sentence(
                cluster_dict_doc, sentence_start, sentence_end)

            # TODO(dwadden) too many outputs. Re-write as a dictionary.
            # Make span indices relative to sentence instead of document.
            ner_dict, relation_dict, cluster_dict, trigger_dict, argument_dict, syntax_dict, children_dict, dep_children_dict, \
            tf_dict \
                = format_label_fields(sentence, ner, relations, cluster_tmp, events, sentence_start, tree,
                                      False, dep, tf, ['F1'])
            sentence_start += len(sentence)
            instance = self._dataset_reader.text_to_instance(
                sentence, ner_dict, relation_dict, cluster_dict, trigger_dict, argument_dict,
                doc_key, dataset, sentence_num, groups, start_ix, end_ix, tree, syntax_dict, children_dict,
                dep_children_dict, tf_dict)
            outputs_one_instance = self.predict_instance(instance)
            outputs.append(outputs_one_instance)

        return outputs
        # instance = self._dataset_reader.text_to_instance(title=title, abstract=abstract)
        #
        # # label_dict will be like {0: "ACL", 1: "AI", ...}
        # label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # # Convert it to list ["ACL", "AI", ...]
        # all_labels = [label_dict[i] for i in range(len(label_dict))]
        #
        # return {"instance": self.predict_instance(instance), "all_labels": all_labels}