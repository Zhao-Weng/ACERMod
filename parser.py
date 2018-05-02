import numpy as np
import torch
import pdb


class Parser:
	def __init__(self, arg):
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
		self.states = torch.from_numpy(np.array(states))
		self.actions = torch.from_numpy(np.array(actions))
		self.rewards = torch.from_numpy(np.array(rewards))
		pdb.set_trace()


if __name__ == '__main__':
	parser = Parser("RLDataset.csv")