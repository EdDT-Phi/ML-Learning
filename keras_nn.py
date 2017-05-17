import random
import numpy as np
import pdb

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from gridworld import initGrid, makeMove, getReward, dispGrid, initGridPlayer, initGridRand

model = Sequential()
model.add(Dense(164, input_shape=(64,), kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
# model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Dense(4, kernel_initializer='lecun_uniform'))
model.add(Activation('linear')) # linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


epochs = 100000
gamma = 0.95 # since it may take several moves to goal, making gamma high
epsilon = 1
won = 0
for i in range(epochs):
	state = initGridRand()
	# while game still in progress
	if i % 1000 == 0 and i != 0:
		print(f'epoch {i/1000}, won {won}')
		won = 0
	
	while True:
		# We are in state S
		# Let's run our Q function on S to get Q values for all possible actions
		qval = model.predict(state.reshape(1,64), batch_size=1)
		if (random.random() < epsilon): # choose random action
			action = np.random.randint(0,4)
		else: # choose best action from Q(s,a) values
			action = (np.argmax(qval))
		
		# Take action, observe new state S'
		new_state = makeMove(state, action)
		
		# Observe reward
		reward = getReward(new_state)
	   
		# Get max_Q(S',a)
		newQ = model.predict(new_state.reshape(1,64), batch_size=1)
		maxQ = np.max(newQ)
		y = np.zeros((1,4))
		y[:] = qval[:]
		if reward == -1: # non-terminal state
			update = (reward + (gamma * maxQ))
		else: # terminal state
			update = reward
		y[0][action] = update #target output
		# print("Game #: %s" % (i,))
		model.fit(state.reshape(1,64), y, batch_size=1, epochs=1, verbose=0)
		state = new_state
		if reward != -1:
			won += 1 if reward == 10 else 0
			break

	if epsilon > 0.1:
		epsilon -= (1/epochs)

def testAlgo(init=0):
	i = 0
	if init==0:
		state = initGrid()
	elif init==1:
		state = initGridPlayer()
	elif init==2:
		state = initGridRand()


	str_to_print = 'Initial State:'
	str_to_print += '\n%s' % dispGrid(state)

	# while game still in progress
	for i in range(10):
		qval = model.predict(state.reshape(1,64), batch_size=1)
		str_to_print += '\n%s' % str(qval)
		action = (np.argmax(qval)) #take action with highest Q-value
		
		new_state = makeMove(state, action)
		reward = getReward(new_state)
		state = new_state
		
		str_to_print += '\n%s' % 'Move #: %s; Taking action: %s' % (i, action)
		str_to_print += '\n%s' % dispGrid(state)
		
		if reward != -1 and reward != -5:
			if reward == -10:
				print(str_to_print + '\n%s' % "Reward: %s" % (reward,))
			return reward
	print(str_to_print + '\n%s' % "Game lost; too many moves.")
	return -10

pdb.set_trace()