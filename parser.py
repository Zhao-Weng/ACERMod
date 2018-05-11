import numpy as np
import pdb
import random

import torch
from torch import nn
from torch.autograd import Variable
from memory import EpisodicReplayMemory
from model import ActorCritic

STATE_SPACE = 22 * 5 + 1
ACTION_SPACE = 22 * 3 * 3
NUM_LAYERS = 2

def isinteger(a):
    try:
        int(a)
        return True
    except ValueError:
        return False


class Parser:
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []

		self.state = []
		self.hm = {}
		self.memory = []
		self.output = []


	def parseOne(self, arg):
		i = 0
		states = []
		actions = []
		rewards = []
		state = []
		action = []
		for line in open(arg):
			if (i > 0):
				curRow = line.split(',')
				curRow = np.array(list(map(float, curRow)))
				if (i % 3 == 1):
					state = np.array(curRow[0:8])
					numpyAction = np.array(curRow[8:12])
					actions.append(numpyAction)
					rewards.append(curRow[12])
				else:
					state = np.concatenate((state, curRow), axis=0)
					if (i % 3 == 0):
						numpyState = np.array(state)
						states.append(numpyState)
			i += 1
		self.states = np.array(states)
		self.actions = torch.FloatTensor(np.array(actions))
		self.rewards = torch.FloatTensor(np.array(rewards))


	def parseInit(self, arg):
		state = []
		i = 0
		for line in open(arg):
			if (i > 0):
				curRow = line.split(',')
				self.hm[i] = curRow[1]
				curRow = curRow[0:1] + curRow[2:] + [0] 
				
				curRow = (list(map(float, curRow)))
				state = state + curRow
			i += 1
		self.state = state


	def generateRandomDataset(self):
		hidden_size = 32
		memory_capacity = 1000
		max_episode_length = 10
		model = ActorCritic(STATE_SPACE, ACTION_SPACE, hidden_size, NUM_LAYERS)
		minLen = 5
		maxLen = 10
		numNode = 22   # [nodeId, bubbleSiz, bubbleScale]
		bubbleSize = 3 
		bubbleScale = 3

		self.memory = EpisodicReplayMemory(memory_capacity, max_episode_length)
		for i in range(1, 101):
			hx = Variable(torch.zeros(NUM_LAYERS, 1, hidden_size))
			# print(hx)
			cx = Variable(torch.zeros(NUM_LAYERS, 1, hidden_size))
			state = self.state + [0]
			rand = random.randint(minLen, maxLen)
			reward = random.uniform(0,10)
			tensorState = torch.zeros(1, STATE_SPACE)
			for timestep in range(1, rand):  # each episode
				question = random.randint(1, numNode) 
				state = state[0: -1] + [question]
				tensorState = torch.FloatTensor(np.array(state)).view(1, STATE_SPACE) # new state
				randIdx = random.randint(0, bubbleSize * bubbleScale - 1)
				action = []
				action.append(question)
				for k in range(2):
					action.append(randIdx % bubbleSize)
					randIdx = randIdx // bubbleSize
				
				policy, _, _, (hx, cx) = model(Variable(tensorState), (hx, cx))
				actionSingleVal = (action[0] - 1) * bubbleScale * bubbleSize + action[1] * bubbleScale + action[2]
				self.memory.append(tensorState, actionSingleVal, reward, policy.data)
				self.output.append(['trial_' + str(i) + '_' + str(timestep), 'Show me ' + self.hm[question], action])
				state[(action[0] - 1) * 5 + 4] = 1

			self.output.append([])
			self.memory.append(tensorState, None, None, None)


	def writeToFile(self, arg):
		file = open(arg,'w')
		file.write('timestep,question ,action_node_id, bubble_size, bubble_scale\n')
		for item in self.output:
			if (len(item) > 0): # not finished for current episode
				for i in range(len(item)- 1):
					file.write(item[i] + ',')
				# pdb.set_trace()
				for feature in item[-1]:
					file.write(str(feature) + ',')
				#file.write('"' + str(item[-1]) + '"') # last action list
			else:
				for i in range(5): # fill 50 in the empty line
					file.write(str(50) + ',')
			file.write('\n')

	# def writeBackMemory(self, arg):




if __name__ == '__main__':
	parser = Parser()
	parser.parseInit('state.csv')
	parser.generateRandomDataset()
	parser.writeToFile('output.csv')
	pdb.set_trace()
