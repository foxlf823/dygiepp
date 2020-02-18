
import json
from stanfordcorenlp import StanfordCoreNLP
from bs4 import BeautifulSoup as BS
from xml.etree import ElementTree

train_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/train.txt'
dev_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/dev.txt'
test_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/test.txt'

# output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json'

rel_directed = False
# output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json-directed'

only_match = False
# output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json-matched'

verbose = False

ctake_input_dir = '/Users/feili/tools/apache-ctakes-4.0.0/clef_input'
ctake_output_dir = '/Users/feili/tools/apache-ctakes-4.0.0/clef_output'
# output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json-ctake'
# output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json-ctake-matched'

use_dep = False
# output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json-dep'

use_tree_feature = True
output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json-tf'
MAX_DEPTH = 3
MAX_PATH = 5

# ALL_INSTANCES,
# ALL_WITH_ENTITIES,
# ONLY_CONTIGUOUS,
# CONTAIN_DISCONTIGUOUS,
# NO_DISCONTIGUOUS,
instanceFilter = 'CONTAIN_DISCONTIGUOUS'

import re
pattern1 = r'([-])'

class Tree(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.match = True

        self.leaf_nodes = []
        self.leaf_nodes_idx = []
        for idx, node in enumerate(self.nodes):
            # if node.data != "":
            if len(node.children) == 0 and node.data != "":  # some leaf nodes don't correspond to tokens
                self.leaf_nodes.append(node)
                self.leaf_nodes_idx.append(idx)

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
    if isinstance(input[-1], str):
        if tokenized:
            tokens = [tok for tok in re.split(pattern1, input[1])
                      if tok not in ["", " "]]
            for token in tokens:
                node = Node(parent_idx, input[0], token)
                nodes.append(node)
                nodes[parent_idx].children.append(len(nodes) - 1)
        else:
            node = Node(parent_idx, input[0], input[1])
            nodes.append(node)
            nodes[parent_idx].children.append(len(nodes) - 1)
    else:
        node = Node(parent_idx, input[0], "")
        nodes.append(node)
        if parent_idx >= 0:
            nodes[parent_idx].children.append(len(nodes) - 1)
        new_parent_idx = len(nodes) - 1
        for child in input[1:]:
            get_node(child, nodes, new_parent_idx, tokenized)

def readTree(text, ind, verbose=False):
    """The basic idea here is to represent the file contents as a long string
    and iterate through it character-by-character (the 'ind' variable
    points to the current character). Whenever we get to a new tree,
    we call the function again (recursively) to read it in."""
    # if verbose:
    #     print("Reading new subtree", text[ind:][:10])

    # consume any spaces before the tree
    while text[ind].isspace():
        ind += 1

    if text[ind] == "(":
        # if verbose:
        #     print("Found open paren")
        tree = []
        ind += 1

        # record the label after the paren
        label = ""
        while not text[ind].isspace() and text != "(":
            label += text[ind]
            ind += 1

        tree.append(label)
        # if verbose:
        #     print("Read in label:", label)

        # read in all subtrees until right paren
        subtree = True
        while subtree:
            # if this call finds only the right paren it'll return False
            subtree, ind = readTree(text, ind, verbose=verbose)
            if subtree:
                tree.append(subtree)

        # consume the right paren itself
        ind += 1
        assert(text[ind] == ")")
        ind += 1

        # if verbose:
        #     print("End of tree", tree)

        return tree, ind

    elif text[ind] == ")":
        # there is no subtree here; this is the end paren of the parent tree
        # which we should not consume
        ind -= 1
        return False, ind

    else:
        # the subtree is just a terminal (a word)
        word = ""
        while not text[ind].isspace() and text[ind] != ")":
            word += text[ind]
            ind += 1

        # if verbose:
        #     print("Read in word:", word)

        return word, ind


def getDepTree(tokens, edges_list):
    nodes = []
    nodes.append(Node(-1, "@NA@", "ROOT"))
    nodes[0].span = [0, len(tokens)-1]
    for token_idx, token_dict in enumerate(tokens):
        node = Node(-2, "@NA@", token_dict['word'])
        node.span = [token_idx, token_idx]
        nodes.append(node)

    for edge in edges_list:
        nodes[edge['governor']].children.append(edge['dependent'])
        nodes[edge['dependent']].parent = edge['governor']

    # check
    for node in nodes:
        assert node.parent != -2
        assert len(set(node.children)) == len(node.children)
    return nodes

# only store adjacent relations
def getDepTree1(tokens, edges_list):
    nodes = [[] for token in tokens]
    nodes_dict = {"nodes":nodes}

    for edge in edges_list:
        if edge['dep'] == 'ROOT':
            continue
        if edge['dependent']-1 not in nodes[edge['governor']-1]:
            nodes[edge['governor']-1].append(edge['dependent']-1)
        if edge['governor']-1 not in nodes[edge['dependent']-1]:
            nodes[edge['dependent']-1].append(edge['governor']-1)

    return nodes_dict

def getPathToRoot(token, token_idx, tree):
    tmp = token
    # path_to_root = [token_idx]
    path_to_root = []
    while tmp.parent != -1:
        path_to_root.append(tmp.parent)
        tmp = tree.nodes[tmp.parent]
    return path_to_root

def getLowestCommonAncestor(token_i_path, token_j_path):
    lca = -1
    for i, node_idx_i in enumerate(token_i_path):
        for j, node_idx_j in enumerate(token_j_path):
            if node_idx_i == node_idx_j:
                lca = node_idx_i
                break
        if lca != -1:
            break
    assert lca != -1
    return i, j, lca

def getTreeFeature_Path(token_i, token_i_idx, token_j, token_j_idx, tree):
    token_i_path = getPathToRoot(token_i, token_i_idx, tree)
    token_j_path = getPathToRoot(token_j, token_j_idx, tree)
    i, j, lca = getLowestCommonAncestor(token_i_path, token_j_path)

    # ret = []
    # ii = 0
    # while ii < i:
    #     ret.append(tree.nodes[token_i_path[ii]].cat)
    #     ii += 1
    # ret.append(tree.nodes[lca].cat)
    # jj = j - 1
    # while jj >= 0:
    #     ret.append(tree.nodes[token_j_path[jj]].cat)
    #     jj -= 1
    # ret = '-'.join(ret)
    # return ret

    ret = []
    ii = 0
    jj = 0
    i_exhaust = False
    j_exhaust = False
    while (not i_exhaust) or (not j_exhaust):

        if ii < i:
            ret.append(tree.nodes[token_i_path[ii]].cat)
            ii += 1
        else:
            i_exhaust = True

        if len(ret) >= MAX_PATH:
            break

        if jj < j:
            ret.append(tree.nodes[token_j_path[jj]].cat)
            jj += 1
        else:
            j_exhaust = True

        if len(ret) >= MAX_PATH:
            break

    if len(ret) < MAX_PATH:
        ret.append(tree.nodes[lca].cat)

    ret = '-'.join(ret)
    return ret


def getTreeFeature_LcaRootSyntax(token_i, token_i_idx, token_j, token_j_idx, tree):
    token_i_path = getPathToRoot(token_i, token_i_idx, tree)
    token_j_path = getPathToRoot(token_j, token_j_idx, tree)
    i, j, lca = getLowestCommonAncestor(token_i_path, token_j_path)

    return tree.nodes[lca].cat

def getTreeFeature_LcaLeftDepth(token_i, token_i_idx, token_j, token_j_idx, tree):
    token_i_path = getPathToRoot(token_i, token_i_idx, tree)
    token_j_path = getPathToRoot(token_j, token_j_idx, tree)
    i, j, lca = getLowestCommonAncestor(token_i_path, token_j_path)

    if i > MAX_DEPTH:
        i = MAX_DEPTH
    return str(i)

def getTreeFeature_LcaRightDepth(token_i, token_i_idx, token_j, token_j_idx, tree):
    token_i_path = getPathToRoot(token_i, token_i_idx, tree)
    token_j_path = getPathToRoot(token_j, token_j_idx, tree)
    i, j, lca = getLowestCommonAncestor(token_i_path, token_j_path)

    if j > MAX_DEPTH:
        j = MAX_DEPTH
    return str(j)

# i is the idx in tree.leaf_nodes, while token_i_idx is the idx in tree.nodes
def getTreeFeature_LcaMatch(token_i_leaf_idx, token_i, token_i_idx, token_j_leaf_idx, token_j, token_j_idx, tree):
    token_i_path = getPathToRoot(token_i, token_i_idx, tree)
    token_j_path = getPathToRoot(token_j, token_j_idx, tree)
    i, j, lca = getLowestCommonAncestor(token_i_path, token_j_path)

    span_left_token_idx = tree.nodes[lca].span[0]
    span_right_token_idx = tree.nodes[lca].span[1]

    left_leaf_idx = token_j_leaf_idx if token_i_leaf_idx > token_j_leaf_idx else token_i_leaf_idx
    right_leaf_idx = token_i_leaf_idx if token_i_leaf_idx > token_j_leaf_idx else token_j_leaf_idx

    if span_left_token_idx == left_leaf_idx and span_right_token_idx == right_leaf_idx:
        return 'm' # match
    elif span_left_token_idx == left_leaf_idx and span_right_token_idx != right_leaf_idx:
        return 'lm' # left match
    elif span_left_token_idx != left_leaf_idx and span_right_token_idx == right_leaf_idx:
        return 'rm' # right match
    else:
        return 'nm' # not match

def getTreeFeatures(tree):
    tree_features = {}
    if not tree.match:
        return tree_features
    # list_list, denotes the relation between token_i and token_j
    # e.g., given a sentence t1 t2 t3, return [[t1_t1, t1_t2, t1_t3],[t2_t1, t2_t2, t2_t3],[t3 ...]]
    tree_features['F1'] = []
    tree_features['F2'] = []
    tree_features['F3'] = []
    tree_features['F4'] = []
    tree_features['F5'] = []
    for i, (token_i, token_i_idx) in enumerate(zip(tree.leaf_nodes, tree.leaf_nodes_idx)):
        f1, f2, f3, f4, f5 = [], [], [], [], []
        for j, (token_j, token_j_idx) in enumerate(zip(tree.leaf_nodes, tree.leaf_nodes_idx)):
            if i == j:
                f1.append('self')
                f2.append('self')
                f3.append('self')
                f4.append('self')
                f5.append('self')
            else:
                f1.append(getTreeFeature_Path(token_i, token_i_idx, token_j, token_j_idx, tree))
                f2.append(getTreeFeature_LcaRootSyntax(token_i, token_i_idx, token_j, token_j_idx, tree)) # lowest common ancestor
                f3.append(getTreeFeature_LcaLeftDepth(token_i, token_i_idx, token_j, token_j_idx, tree))
                f4.append(getTreeFeature_LcaRightDepth(token_i, token_i_idx, token_j, token_j_idx, tree))
                f5.append(getTreeFeature_LcaMatch(i, token_i, token_i_idx, j, token_j, token_j_idx, tree))
        tree_features['F1'].append(f1)
        tree_features['F2'].append(f2)
        tree_features['F3'].append(f3)
        tree_features['F4'].append(f4)
        tree_features['F5'].append(f5)

    return tree_features


def hasDiscontiguousEntity(instance):
    entities = instance['entities']
    for entity in entities:
        if len(entity['span']) > 1:
            return True
    return False


def read_file(file_path, instanceFilter):
    filtered_instances = []
    fp = open(file_path, 'r', encoding='utf-8')
    for line in fp:
        line = line.strip()
        if len(line) == 0:
            continue

        instance = json.loads(line)
        if instanceFilter == 'CONTAIN_DISCONTIGUOUS':
            if not hasDiscontiguousEntity(instance):
                continue
        else:
            pass

        filtered_instances.append(instance)

    fp.close()
    return filtered_instances

def do_statistics(instances):
    entity_1seg = 0
    entity_2seg = 0
    entity_3seg = 0
    stats = dict(span=0, span_common=0)
    for instance in instances:
        for entity in instance['entities']:
            if len(entity['span']) == 1:
                entity_1seg += 1
            elif len(entity['span']) == 2:
                entity_2seg += 1
            else:
                entity_3seg += 1

            for span in entity['span']:
                start_end = span.split(',')
                start = int(start_end[0])
                end = int(start_end[1])
                stats['span'] += 1
                if end-start+1 <= 6:
                    stats['span_common'] += 1

    print("sentence number: {}".format(len(instances)))
    print("1 segment entity: {}".format(entity_1seg))
    print("2 segment entity: {}".format(entity_2seg))
    print("3 segment entity: {}".format(entity_3seg))
    print(stats)

import os
def transfer_into_dygie(instances, output_file):
    stats_treebank = dict(sent_notree=0, sent_mismatches=0, sent_matches=0)
    fp = open(output_file, 'w')
    for idx, instance in enumerate(instances):
        doc = {}
        doc['doc_key'] = instance['doc']+"_"+str(instance['start'])+"_"+str(instance['end'])
        doc['sentences'] = []
        doc['sentences'].append(instance['tokens'])
        doc['ner'] = []
        ner_for_this_sentence = []
        doc['relations'] = []
        relation_for_this_sentence = []
        for entity in instance['entities']:
            for span in entity['span']:
                start = int(span.split(',')[0])
                end = int(span.split(',')[1])
                entity_output = [start, end, entity['type']]
                ner_for_this_sentence.append(entity_output)

            n_spans = len(entity['span'])
            if rel_directed:
                candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i < j]
            else:
                candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i!=j] # undirected relations
            for i, j in candidate_indices:
                arg1_start = int(entity['span'][i].split(',')[0])
                arg1_end = int(entity['span'][i].split(',')[1])
                arg2_start = int(entity['span'][j].split(',')[0])
                arg2_end = int(entity['span'][j].split(',')[1])
                relation_for_this_sentence.append([arg1_start, arg1_end, arg2_start, arg2_end, "Combined"])

        doc['ner'].append(ner_for_this_sentence)
        doc['relations'].append(relation_for_this_sentence)

        doc['trees'] = []
        if use_dep:
            doc['dep'] = []
        if use_tree_feature:
            doc['tf'] = []
        try:
            nlp_res_raw = nlp.annotate(' '.join(instance['tokens']), properties={'annotators': 'tokenize,ssplit,pos,parse'})
            nlp_res = json.loads(nlp_res_raw)
        except Exception as e:
            if verbose:
                print('[Warning] StanfordCore Timeout: ', instance['tokens'])
            if only_match:
                continue
            else:
                doc['trees'].append({})
                stats_treebank['sent_notree'] += 1
                if use_dep:
                    doc['dep'].append({})
                if use_tree_feature:
                    doc['tf'].append({})
                fp.write(json.dumps(doc) + "\n")
                continue

        if len(nlp_res['sentences']) >= 2:
            if verbose:
                print('[Warning] sentence segmentation of StandfordCoreNLP do not match: ', instance['tokens'])
            if only_match:
                continue
            else:
                doc['trees'].append({})
                stats_treebank['sent_notree'] += 1
                if use_dep:
                    doc['dep'].append({})
                if use_tree_feature:
                    doc['tf'].append({})
                fp.write(json.dumps(doc) + "\n")
                continue

        tree, _ = readTree(nlp_res['sentences'][0]['parse'], 0)
        nodes = []
        get_node(tree, nodes, -1, False)
        tree = Tree(nodes)
        tree.get_span_for_leaf_node(instance['tokens'])
        if not tree.match:
            if verbose:
                print("sent mismatch, doc: {}, original sentence: {}, tree sentence {}".format(doc['doc_key'],
                                                                                               instance['tokens'],
                                                                                               tree.show_leaf_node()))
            if only_match:
                continue
            else:
                stats_treebank['sent_mismatches'] += 1
        else:
            stats_treebank['sent_matches'] += 1
        tree.get_span_for_node(instance['tokens'])
        doc['trees'].append(tree.to_json())
        if use_dep:
            # dep_nodes = getDepTree(nlp_res['sentences'][0]['tokens'], nlp_res['sentences'][0]['basicDependencies'])
            # dep_tree = Tree(dep_nodes)
            # doc['dep'].append(dep_tree.to_json())
            dep_nodes = getDepTree1(nlp_res['sentences'][0]['tokens'], nlp_res['sentences'][0]['basicDependencies'])
            doc['dep'].append(dep_nodes)
        if use_tree_feature:
            tree_features = getTreeFeatures(tree)
            doc['tf'].append(tree_features)

        fp.write(json.dumps(doc)+"\n")

    fp.close()
    print(stats_treebank)

def prepare_data_for_ctake(instances, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, instance in enumerate(instances):

        file_name = instance['doc'][instance['doc'].rfind('/')+1:] + "_" + str(instance['start']) + "_" + str(instance['end'])

        fp = open(os.path.join(output_dir, file_name), 'w')
        fp.write(instance['text']+"\n")
        fp.close()

def transfer_ctake_data_into_dygie(instances, ctake_dir, output_file):
    stats_treebank = dict(sent_notree=0, sent_mismatches=0, sent_matches=0)
    fp = open(output_file, 'w')
    for idx, instance in enumerate(instances):
        doc = {}
        doc['doc_key'] = instance['doc'] + "_" + str(instance['start']) + "_" + str(instance['end'])
        doc['sentences'] = []
        doc['sentences'].append(instance['tokens'])
        doc['ner'] = []
        ner_for_this_sentence = []
        doc['relations'] = []
        relation_for_this_sentence = []
        for entity in instance['entities']:
            for span in entity['span']:
                start = int(span.split(',')[0])
                end = int(span.split(',')[1])
                entity_output = [start, end, entity['type']]
                ner_for_this_sentence.append(entity_output)

            n_spans = len(entity['span'])
            if rel_directed:
                candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i < j]
            else:
                candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if
                                     i != j]  # undirected relations
            for i, j in candidate_indices:
                arg1_start = int(entity['span'][i].split(',')[0])
                arg1_end = int(entity['span'][i].split(',')[1])
                arg2_start = int(entity['span'][j].split(',')[0])
                arg2_end = int(entity['span'][j].split(',')[1])
                relation_for_this_sentence.append([arg1_start, arg1_end, arg2_start, arg2_end, "Combined"])

        doc['ner'].append(ner_for_this_sentence)
        doc['relations'].append(relation_for_this_sentence)

        doc['trees'] = []
        ctake_file_name = instance['doc'][instance['doc'].rfind('/')+1:] + "_" + str(instance['start']) + "_" + str(instance['end'])

        tree = ElementTree.parse(os.path.join(ctake_dir, ctake_file_name+".xml"))
        root = tree.getroot()
        tree_sentences = []
        for child in root:
            if child.tag == 'org.apache.ctakes.typesystem.type.syntax.TopTreebankNode':
                tree_sentences.append(child.attrib['treebankParse'])

        if len(tree_sentences) != 1:
            if verbose:
                print('[Warning] sentence segmentation of ctake do not match: ', instance['tokens'])
            if only_match:
                continue
            else:
                doc['trees'].append({})
                stats_treebank['sent_notree'] += 1
        else:
            tree, _ = readTree(tree_sentences[0], 0)
            nodes = []
            get_node(tree, nodes, -1, False)
            tree = Tree(nodes)
            tree.get_span_for_leaf_node(instance['tokens'])
            if not tree.match:
                if verbose:
                    print("sent mismatch, doc: {}, original sentence: {}, tree sentence {}".format(doc['doc_key'],
                                                                                                   instance['tokens'],
                                                                                                   tree.show_leaf_node()))
                if only_match:
                    continue
                else:
                    stats_treebank['sent_mismatches'] += 1
            else:
                stats_treebank['sent_matches'] += 1
            tree.get_span_for_node(instance['tokens'])
            doc['trees'].append(tree.to_json())

        fp.write(json.dumps(doc) + "\n")

    fp.close()
    print(stats_treebank)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with StanfordCoreNLP('./stanford-corenlp-full-2018-10-05', memory='8g', timeout=60000) as nlp:

        filtered_instances = read_file(train_file, instanceFilter)
        # do_statistics(filtered_instances)
        transfer_into_dygie(filtered_instances, os.path.join(output_dir, 'train.json'))
        # prepare_data_for_ctake(filtered_instances, os.path.join(ctake_input_dir, 'train'))
        # transfer_ctake_data_into_dygie(filtered_instances, os.path.join(ctake_output_dir, 'train'), os.path.join(output_dir, 'train.json'))

        filtered_instances = read_file(dev_file, instanceFilter)
        # do_statistics(filtered_instances)
        transfer_into_dygie(filtered_instances, os.path.join(output_dir, 'dev.json'))
        # prepare_data_for_ctake(filtered_instances, os.path.join(ctake_input_dir, 'dev'))
        # transfer_ctake_data_into_dygie(filtered_instances, os.path.join(ctake_output_dir, 'dev'),
        #                                os.path.join(output_dir, 'dev.json'))

        filtered_instances = read_file(test_file, instanceFilter)
        # do_statistics(filtered_instances)
        transfer_into_dygie(filtered_instances, os.path.join(output_dir, 'test.json'))
        # prepare_data_for_ctake(filtered_instances, os.path.join(ctake_input_dir, 'test'))
        # transfer_ctake_data_into_dygie(filtered_instances, os.path.join(ctake_output_dir, 'test'),
        #                                os.path.join(output_dir, 'test.json'))

    pass