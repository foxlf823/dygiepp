
from allennlp.nn import util
import torch
from allennlp.modules.time_distributed import TimeDistributed
from torch.nn.parameter import Parameter
import copy

# a = torch.LongTensor([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
# mask = torch.LongTensor([[[1,1,1],[1,1,0]],[[1,1,0],[1,0,0]]])
# b = util.masked_max(a, mask, 1, True)
#
# a = torch.LongTensor([[[1,2,3],[4,5,0]],[[1,2,0],[4,0,0]]])
# mask = torch.LongTensor([[[1,1,1],[1,1,0]],[[1,1,0],[1,0,0]]])
# b = util.masked_max(a, mask, 1, True)
#
# a = torch.LongTensor([[[1,2,3],[4,5,6]],[[1,-2,3],[4,5,6]]])
# mask = torch.LongTensor([[[1,1,1],[1,1,0]],[[1,1,0],[1,0,0]]])
# b = util.masked_max(a, mask, 1, False)

# sequence_tensor = torch.FloatTensor([[[7,8,9,10],[4,5,6,10],[1,2,3,10]],[[1,-2,3,10],[4,5,6,10],[-1,-2,-3,10]]])
# span_indices = torch.LongTensor([[[0,0], [0,1], [0,2], [1,2]],[[0,0], [0,1], [0,2], [1,2]]])
# _global_attention = TimeDistributed(torch.nn.Linear(3, 1))
# _conv = torch.nn.Conv1d(3, 3, kernel_size=3, padding=1)
# _input_dim = 4
# _start_sentinel = Parameter(torch.randn([1, 1, int(_input_dim / 2)]))
# _end_sentinel = Parameter(torch.randn([1, 1, int(_input_dim / 2)]))
# _forward_combination = "y"
# _backward_combination = "y"


# line = 'asdf  fjdk; afed, fjek,asdf, foo,'
# import re
# # print(re.split(r'([;,\s]\s*)', line))
# print(line.split(' '))

# a = set()
# a.add((7,7))
# a.add((4,4))
# b = set()
# b.add((3,3))
# b.add((4,4))
#
#
# print(a == b)

# import dygie
# from allennlp.common.testing import ModelTestCase
# class TestPaperClassifierPredictor(TestCase):
#     def test_uses_named_inputs(self):
#         inputs = {
#             "title": "Interferring Discourse Relations in Context",
#             "paperAbstract": (
#                     "We investigate various contextual effects on text "
#                     "interpretation, and account for them by providing "
#                     "contextual constraints in a logical theory of text "
#                     "interpretation. On the basis of the way these constraints "
#                     "interact with the other knowledge sources, we draw some "
#                     "general conclusions about the role of domain-specific "
#                     "information, top-down and bottom-up discourse information "
#                     "flow, and the usefulness of formalisation in discourse theory."
#             )
#         }
#
#         archive = load_archive('tests/fixtures/model.tar.gz')
#         predictor = Predictor.from_archive(archive, 'paper-classifier')
#
#         result = predictor.predict_json(inputs)
#
#         label = result.get("all_labels")
#         assert label in ['AI', 'ML', 'ACL']
#
#         class_probabilities = result.get("class_probabilities")
#         assert class_probabilities is not None
#         assert all(cp > 0 for cp in class_probabilities)
#         assert sum(class_probabilities) == approx(1.0)


# from stanfordcorenlp import StanfordCoreNLP
#
# nlp = StanfordCoreNLP('http://localhost', port=9001)
# sentence = 'UMASS Lowell is a good university.'
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))
# print('Dependency Parsing:', nlp.dependency_parse(sentence))
# nlp.close() # Do not forget to close! The backend server will consume a lot memery.
#
# pass

import numpy as np

x = np.array([3, 1, 2])
print(np.argsort(x))
print(x[np.argsort(x)])
print(np.argsort(-x))
print(x[np.argsort(-x)])
print(x.tolist())