"""
Add coreference information to the GENIA data set.

Also remove blacklisted documents, for which the merge didn't go correctly and off-by-one-errors
were introduced.
"""

from collections import defaultdict

from os import path
import os
import glob
import re
import json
from bs4 import BeautifulSoup as BS
import pandas as pd
import argparse

import shared
import copy


genia_base = "./data/genia"
genia_raw = f"{genia_base}/raw-data"
genia_processed = f"{genia_base}/processed-data"

json_dir = f"{genia_processed}/json-ner"
tree_dir = f"{genia_raw}/GENIA_treebank_v1"
coref_dir = f"{genia_raw}/GENIA_MedCo_coreference_corpus_1.0"
alignment_file = f"{genia_raw}/align/alignment.csv"

alignment = pd.read_csv(alignment_file).set_index("ner")

class Coref(object):
    """Represents a single coreference."""

    def __init__(self, xml, soup_text, sents):
        "Get the text, id, referrent, and text span."
        self.xml = xml
        self.text = xml.text
        self.tokens = [tok for tok in re.split('([ -/,.+])', self.text)
                       if tok not in ["", " "]]
        self.id = xml.attrs["id"]
        # A very small number of corefs have two parents. I'm going to just take the first parent.
        # TODO(dwadden) If time, go back and fix this.
        self.ref = xml.attrs["ref"].split(" ")[0] if "ref" in xml.attrs else None
        self.span = self._get_span(sents, soup_text)
        self.type = xml.attrs["type"] if "type" in xml.attrs else None

    def _get_span(self, sents, soup_text):
        """Get text span of coref. We have inclusive endpoints."""

        spans = shared.find_sub_lists(self.tokens, sents)
        n_matches = len(spans)
        # Case 1: If can't match the span, record and return. This doesn't happen
        # much.
        if n_matches == 0:
            stats["no_matches"] += 1
            return None
        # Case 2: IF there are multiple span matches, go back and look the original
        # XML tag to determine which match we want.
        elif n_matches > 1:
            xml_tag = self.xml.__repr__()
            tmp_ixs = shared.find_sub_lists(list(self.text), list(soup_text))
            text_ixs = []
            # Last character of the match must be a dash or char after must be an
            # escape, else we're not at end of token.
            text_ixs = [ixs for ixs in tmp_ixs if
                        soup_text[ixs[1] + 1] in '([ -/,.+])<' or soup_text[ixs[1]] == "-"]
            if len(text_ixs) != n_matches:
                # If the number of xml tag matches doesn't equal the number of span
                # matches, record and return.
                stats["different_num_matches"] += 1
                return None
            tag_ix = shared.find_sub_lists(list(xml_tag), list(soup_text))
            assert len(tag_ix) == 1
            tag_ix = tag_ix[0]
            text_inside = [x[0] >= tag_ix[0] and x[1] <= tag_ix[1] for x in text_ixs]
            assert sum(text_inside) == 1
            match_ix = text_inside.index(True)
        else:
            match_ix = 0
        stats["successful_matches"] += 1
        span = spans[match_ix]
        return span


class Corefs(object):
    """Holds all corefs and represents relations between them."""

    def __init__(self, soup, sents_flat, coref_types):
        self.coref_types = coref_types
        coref_items = soup.find_all("coref")
        corefs = [Coref(item, soup.__repr__(), sents_flat) for item in coref_items]
        # Put the cluster exemplars first.
        corefs = sorted(corefs, key=lambda coref: coref.ref is None, reverse=True)
        coref_ids = [coref.id for coref in corefs]
        corefs = self._assign_parent_indices(corefs, coref_ids)
        clusters = self._get_coref_clusters(corefs)
        clusters = self._cleanup_coref_clusters(corefs, clusters)
        cluster_spans = self._make_cluster_spans(clusters)
        self.corefs = corefs
        self.clusters = clusters
        self.cluster_spans = cluster_spans

    @staticmethod
    def _assign_parent_indices(corefs, coref_ids):
        """Give each coref the index of it parent in the list of corefs."""
        for coref in corefs:
            if coref.ref is None:
                coref.parent_ix = None
            else:
                coref.parent_ix = coref_ids.index(coref.ref)
        return corefs

    @staticmethod
    def _get_coref_clusters(corefs):
        def get_cluster_assignment(coref):
            ids_so_far = set()
            this_coref = coref
            while this_coref.ref is not None:
                # Condition to prevent self-loops.
                if this_coref.id in ids_so_far or this_coref.id == this_coref.ref:
                    return None
                ids_so_far.add(this_coref.id)
                parent = corefs[this_coref.parent_ix]
                this_coref = parent
            return this_coref.id

        clusters = {None: set()}
        for coref in corefs:
            if coref.ref is None:
                # It's a cluster exemplar
                coref.cluster_assignment = coref.id
                clusters[coref.id] = set([coref])
            else:
                cluster_assignment = get_cluster_assignment(coref)
                coref.cluster_assignment = cluster_assignment
                clusters[cluster_assignment].add(coref)
        return clusters

    def _cleanup_coref_clusters(self, corefs, clusters):
        """
        Remove items that didn't get spans, don't have an allowed coref type, or
        weren't assigned a cluster
        """
        # Remove unassigned corefs.
        _ = clusters.pop(None)
        for coref in corefs:
            # If the referent entity didn't get a span match, remove the cluster.
            if coref.ref is None:
                if coref.span is None:
                    _ = clusters.pop(coref.id)
            # If a referring coref didn't have a span or isn't the right coref type, remove it.
            else:
                if coref.type not in self.coref_types or coref.span is None:
                    # Check to make sure the cluster wasn't already removed.
                    if coref.cluster_assignment in clusters:
                        clusters[coref.cluster_assignment].remove(coref)
        # Now remove singleton clusters.
        # Need to make it a list to avoid `dictionary size changed iteration` error."
        for key in list(clusters.keys()):
            if len(clusters[key]) == 1:
                _ = clusters.pop(key)
        return clusters

    @staticmethod
    def _make_cluster_spans(clusters):
        """Convert to nested list of cluster spans, as in scierc data."""
        res = []
        for key, cluster in clusters.items():
            cluster_spans = []
            for coref in cluster:
                cluster_spans.append(list(coref.span))
            res.append(sorted(cluster_spans))
        return res


class Tree(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.match = True

        self.leaf_nodes = []
        for node in self.nodes:
            # if node.data != "":
            if len(node.children) == 0 and node.data != "":  # some leaf nodes don't correspond to tokens
                self.leaf_nodes.append(node)

    def get_span_for_leaf_node(self, sentence):
        if len(self.leaf_nodes) != len(sentence):
            self.match = False
            for idx, node in enumerate(self.leaf_nodes):
                node.span = [idx, idx]  # inclusive endpoints
        else:
            for idx, (node, text) in enumerate(zip(self.leaf_nodes, sentence)):
                if node.data != text:
                    self.match = False
                node.span = [idx, idx]  # inclusive endpoints


    def get_span_for_node(self, sentence):

        # assert (len(sentence) == len(self.leaf_nodes))
        # for idx, (node, text) in enumerate(zip(self.leaf_nodes, sentence)):
        #     assert (node.data == text)
        #     node.span = [idx, idx]  # inclusive endpoints

        sentence_start, sentence_end = Tree.get_span_for_node_(self.nodes, 0)
        # assert(sentence_start==0)
        # assert(sentence_end==len(sentence)-1)

        for node in self.nodes:
            if node.span[1] == -1: # update the span for Null constituent
                node.span[0] = -1

    def show_leaf_node(self):
        ret = [n.data for n in self.leaf_nodes]
        return ret


    @staticmethod
    def get_span_for_node_(nodes, node_idx):
        node = nodes[node_idx]
        if len(node.children) == 0:
            if node.data != "":
                return node.span[0], node.span[1]
            else:
                node.span = [999, -1]
                return 999, -1
        else:
            span_start = 999
            span_end = -1
            for child_idx in node.children:
                child_span_start, child_span_end = Tree.get_span_for_node_(nodes, child_idx)
                span_start = child_span_start if child_span_start < span_start else span_start
                span_end = child_span_end if child_span_end > span_end else span_end

            # assert(span_start<=span_end)
            node.span = [span_start, span_end]
            return span_start, span_end

    def print_tree(self):
        return Tree._print_node(self.nodes, 0, "", 0)

    @staticmethod
    def _print_node(nodes, node_idx, ret, level):
        node = nodes[node_idx]
        indent = '  '*level
        ret += '\n'+indent+'('+node.cat if node.data == '' else ' ('+node.cat+' '+node.data
        for child_idx in node.children:
            ret = Tree._print_node(nodes, child_idx, ret, level+1)
        ret += ')'
        return ret

    def to_json(self):
        nodes = []
        ret = {'match':self.match, 'nodes': nodes}
        for node in self.nodes:
            nodes.append(node.to_json())
        return ret



class Node(object):
    def __init__(self, parent, cat, data):
        self.parent = parent
        self.cat = cat
        self.data = data
        self.children = []

    def __str__(self):
        return self.cat+"|"+self.data+"|"+str(self.parent)

    def to_json(self):
        return [self.parent, self.children, self.cat, self.data, self.span]

def get_node(input, nodes, parent_idx, tokenized):
    for child in input.children:
        if child.name == 'cons':
            # if "null" in child.attrs:
            #     continue # ignore Null constituent
            node = Node(parent_idx, child.attrs["cat"], "")
            nodes.append(node)
            if parent_idx >= 0:
                nodes[parent_idx].children.append(len(nodes)-1)
            get_node(child, nodes, len(nodes)-1, tokenized)
        elif child.name == 'tok':
            if tokenized:
                tokens = [tok for tok in re.split('([ -/,.+])', child.text)
                           if tok not in ["", " "]]
                for token in tokens:
                    node = Node(parent_idx, child.attrs["cat"], token)
                    nodes.append(node)
                    nodes[parent_idx].children.append(len(nodes) - 1)
            else:
                node = Node(parent_idx, child.attrs["cat"], child.text)
                nodes.append(node)
                nodes[parent_idx].children.append(len(nodes)-1)

def matched(tree, sentence):
    if len(sentence) >= 3 and len(tree.leaf_nodes) >= 3:
        if tree.leaf_nodes[0].data == sentence[0] and tree.leaf_nodes[1].data == sentence[1] and tree.leaf_nodes[2].data == sentence[2] and \
            tree.leaf_nodes[-1].data == sentence[-1] and tree.leaf_nodes[-2].data == sentence[-2] and tree.leaf_nodes[-3].data == sentence[-3]:
            return True
        else:
            return False
    else:
        if tree.leaf_nodes[0].data == sentence[0] and tree.leaf_nodes[1].data == sentence[1] and \
            tree.leaf_nodes[-1].data == sentence[-1] and tree.leaf_nodes[-2].data == sentence[-2]:
            return True
        else:
            return False


def build_tree(soup, doc, tokenized, pmid, medline_id):
    tree_sentences = soup.find_all("sentence")

    # assert(len(tree_sentences)==len(sentences))
    if len(tree_sentences) != len(doc['sentences']):
        print("doc mismatch, term doc: {}, doc len {}, treebank doc: {}, doc len {}".format(pmid, len(doc['sentences']), medline_id, len(tree_sentences)))
        stats_treebank['doc_mismatches'] += 1
    else:
        stats_treebank['doc_matches'] += 1

    trees = []
    start = 0
    for sentence in doc['sentences']:
        b_match = False
        for idx, tree_sentence in enumerate(tree_sentences[start:]):
            nodes = []
            get_node(tree_sentence, nodes, -1, tokenized)
            tree = Tree(nodes)
            if matched(tree, sentence):
                b_match = True
                break
        if b_match:
            tree.get_span_for_leaf_node(sentence)
            if not tree.match:
                print(
                    "sent mismatch, term doc: {}, term sentence: {}, treebank doc: {}, treebank sentence {}".format(
                        pmid, sentence, medline_id, tree.show_leaf_node()))
                stats_treebank['sent_mismatches'] += 1
            else:
                stats_treebank['sent_matches'] += 1
            tree.get_span_for_node(sentence)
            trees.append(tree.to_json())
            start = start + idx + 1
        else:
            print(
                "sent not find tree, term doc: {}, term sentence: {}, treebank doc: {}".format(
                    pmid, sentence, medline_id))
            stats_treebank['sent_notree'] += 1
            trees.append({})


    # for tree_sentence, sentence in zip(tree_sentences, doc['sentences']):
    #     nodes = []
    #     get_node(tree_sentence, nodes, -1, tokenized)
    #     tree = Tree(nodes)
    #     # print(tree.print_tree())
    #     tree.get_span_for_leaf_node(sentence)
    #     if not tree.match:
    #         print("sent mismatch, term doc: {}, term sentence: {}, treebank doc: {}, treebank sentence {}".format(pmid, sentence, medline_id, tree.show_leaf_node()))
    #         stats_treebank['sent_mismatches'] += 1
    #     else:
    #         stats_treebank['sent_matches'] += 1
    #     tree.get_span_for_node(sentence)
    #     trees.append(tree.to_json())

    return trees


def get_excluded():
    "Get list of files that had random off-by-1-errors and will be excluded."
    current_path = os.path.dirname(os.path.realpath(__file__))
    excluded = pd.read_table(f"{current_path}/exclude.txt", header=None, squeeze=True).values
    return excluded

from collections import Counter

def one_fold(fold, coref_types, out_dir, keep_excluded):
    """Add coref field to json, one fold."""
    print("Running fold {0}.".format(fold))
    entity_length = Counter()
    entity_stat = dict(entity=0, entity_common=0)
    excluded = get_excluded()
    with open(path.join(json_dir, "{0}.json".format(fold))) as f_json:
        with open(path.join(out_dir, "{0}.json".format(fold)), "w") as f_out:
            for counter, line in enumerate(f_json):
                doc = json.loads(line)
                for sentence, ner in zip(doc['sentences'], doc['ner']):
                    for e in ner:
                        entity_stat['entity'] += 1
                        entity_length[e[1]-e[0]+1] += 1
                        if e[1] - e[0] + 1 <= 10:
                            entity_stat['entity_common'] += 1
                pmid = int(doc["doc_key"].split("_")[0])
                medline_id = alignment.loc[pmid][0]
                # xml_file = path.join(coref_dir, str(medline_id) + ".xml")
                # sents_flat = shared.flatten(doc["sentences"])
                xml_tree_file = path.join(tree_dir, str(medline_id) + ".xml")
                # with open(xml_file, "r") as f_xml:
                #     soup = BS(f_xml.read(), "lxml")
                #     corefs = Corefs(soup, sents_flat, coref_types)
                # doc["clusters"] = corefs.cluster_spans
                with open(xml_tree_file, 'r') as f_xml:
                    soup = BS(f_xml.read(), "lxml")
                    trees = build_tree(soup, doc, True, pmid, medline_id)
                doc["trees"] = trees
                # Save unless it's bad and we're excluding bad documents.
                if keep_excluded or doc["doc_key"] not in excluded:
                    f_out.write(json.dumps(doc) + "\n")
    print(entity_length)
    print(entity_stat)

def get_clusters(coref_types, out_dir, keep_excluded):
    """Add coref to json, filtering to only keep coref roots and `coref_types`."""
    global stats
    stats = dict(no_matches=0, successful_matches=0, different_num_matches=0)
    global stats_treebank
    stats_treebank = dict(doc_mismatches=0, doc_matches=0, sent_mismatches=0, sent_matches=0, sent_notree=0)
    for fold in ["train", "dev", "test"]:
        one_fold(fold, coref_types, out_dir, keep_excluded)
    print(stats)
    print(stats_treebank)


def main():
    # get_coref_types()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ident-only", action="store_true",
                        help="If true, only do `IDENT` coreferences.")
    parser.add_argument("--keep-excluded", action="store_true",
                        help="If true, keep training docs that were excluded due to off-by-1 errors.")
    args = parser.parse_args()
    if args.ident_only:
        coref_types = ["IDENT"]
    else:
        coref_types = [
            "IDENT",
            "NONE",
            "RELAT",
            "PRON",
            "APPOS",
            "OTHER",
            "PART-WHOLE",
            "WHOLE-PART"
        ]

    out_dir = f"{genia_processed}/json-treebank"

    if not path.exists(out_dir):
        os.mkdir(out_dir)

    get_clusters(coref_types, out_dir, args.keep_excluded)


if __name__ == '__main__':
    main()
