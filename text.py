
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

sequence_tensor = torch.FloatTensor([[[7,8,9,10],[4,5,6,10],[1,2,3,10]],[[1,-2,3,10],[4,5,6,10],[-1,-2,-3,10]]])
span_indices = torch.LongTensor([[[0,0], [0,1], [0,2], [1,2]],[[0,0], [0,1], [0,2], [1,2]]])
_global_attention = TimeDistributed(torch.nn.Linear(3, 1))
_conv = torch.nn.Conv1d(3, 3, kernel_size=3, padding=1)
_input_dim = 4
_start_sentinel = Parameter(torch.randn([1, 1, int(_input_dim / 2)]))
_end_sentinel = Parameter(torch.randn([1, 1, int(_input_dim / 2)]))
_forward_combination = "y"
_backward_combination = "y"


# line = 'asdf  fjdk; afed, fjek,asdf, foo,'
# import re
# # print(re.split(r'([;,\s]\s*)', line))
# print(line.split(' '))

a = set()
a.add((7,7))
a.add((4,4))
b = set()
b.add((3,3))
b.add((4,4))


print(a == b)

pass