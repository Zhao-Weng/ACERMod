# -*- coding: utf-8 -*-
from torch import nn


class ActorCritic(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, num_layers):
    super(ActorCritic, self).__init__()
    self.state_size = state_size
    self.action_size = action_size

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax(dim=1)

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = 2)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, self.action_size)

  def forward(self, x, h):
    x = self.relu(self.fc1(x))
    print(x.size())
    print(h.size())
    hlower, hupper = self.lstm(x, h)  # h is (hidden state, cell state)

    x = hlower[0]

    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)

    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h
