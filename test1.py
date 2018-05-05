import torch
from torch import nn
from torch.autograd import Variable
import pdb

lstm = nn.LSTMCell(4, 4)
h0, c0 = torch.FloatTensor(torch.zeros(1,4)), torch.FloatTensor(torch.zeros(1,4))
x = torch.zeros(1, 4)
b = lstm(x, (h0,c0))

pdb.set_trace()