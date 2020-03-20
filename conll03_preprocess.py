import json
from stanfordcorenlp import StanfordCoreNLP


verbose = False

MAX_DEPTH = 3
MAX_PATH = 5


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
        f1, f2, f3, f4, f5 = [], [], [], [], [],
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

def conll_2_json(conll_path, json_path):
    type = conll_path[conll_path.rfind('/')+1:conll_path.rfind('.')]
    id = 1
    stats_treebank = dict(sent_notree=0, sent_mismatches=0, sent_matches=0)
    in_fp = open(conll_path, 'r')
    out_fp = open(json_path, 'w')
    sentences = []
    ner = []
    relations = []

    for line in in_fp:
        line = line.strip()
        if len(line) == 0:
            if len(sentences) != 0:
                instance = dict(doc_key='', sentences=[], ner=[], relations=[], trees=[], tf=[])
                instance['doc_key'] = type+"_"+str(id)
                id += 1
                instance['sentences'].append(sentences)
                instance['ner'].append(ner)
                instance['relations'].append(relations)

                try:
                    nlp_res_raw = nlp.annotate(' '.join(sentences), properties={'annotators': 'tokenize,ssplit,pos,parse'})
                    nlp_res = json.loads(nlp_res_raw)

                    if len(nlp_res['sentences']) >= 2:
                        if verbose:
                            print('[Warning] sentence segmentation of StandfordCoreNLP do not match: ', sentences)
                        stats_treebank['sent_notree'] += 1

                        instance['trees'].append({})
                        instance['tf'].append({})

                    else:
                        tree, _ = readTree(nlp_res['sentences'][0]['parse'], 0)
                        nodes = []
                        get_node(tree, nodes, -1, False)
                        tree = Tree(nodes)
                        tree.get_span_for_leaf_node(sentences)
                        if not tree.match:
                            if verbose:
                                print("sent mismatch, doc: {}, original sentence: {}, tree sentence {}".format(
                                    instance['doc_key'], sentences, tree.show_leaf_node()))

                            stats_treebank['sent_mismatches'] += 1
                        else:
                            stats_treebank['sent_matches'] += 1

                        tree.get_span_for_node(sentences)
                        instance['trees'].append(tree.to_json())

                        tree_features = getTreeFeatures(tree)
                        instance['tf'].append(tree_features)

                except Exception as e:
                    if verbose:
                        print('[Warning] StanfordCore Timeout: ', sentences)
                    stats_treebank['sent_notree'] += 1

                    instance['trees'].append({})
                    instance['tf'].append({})

                out_fp.write(json.dumps(instance)+"\n")
                sentences = []
                ner = []
                relations = []

            continue
        columns = line.split(' ')
        label = columns[1]
        if label[0] == 'B' or label[0] == 'S':
            entity = [len(sentences), len(sentences), label[label.find('-')+1:]]
            ner.append(entity)
        elif label[0] == 'I' or label[0] == 'E':
            entity = ner[-1]
            entity[1] = len(sentences)
        sentences.append(columns[0])

    in_fp.close()
    out_fp.close()
    print(stats_treebank)

if __name__ == "__main__":
    # with StanfordCoreNLP('./files_for_discontinuous_ner/stanford-corenlp-full-2018-10-05', memory='8g', timeout=60000) as nlp:
    #     conll_2_json('./data/conll03/train.txt', './data/conll03/train.json')
    #     conll_2_json('./data/conll03/valid.txt', './data/conll03/dev.json')
    #     conll_2_json('./data/conll03/test.txt', './data/conll03/test.json')

    pass