import numpy as np
from scipy import stats
from random import random
import matplotlib.pyplot as plt
# %matplotlib inline

n = 10
arms = np.random.rand(n)
eps = 0.1

def reward(prob):
	reward = 0;
	for i in range(10):
		if random() < prob:
			reward += 1
	return reward

#initialize memory array; has 1 row defaulted to random action index
av = np.array([np.random.randint(0,(n+1)), 0]).reshape(1,2) #av = action-value

#greedy method to select best arm based on memory array (historical results)
def bestArm(a):
	bestArm = 0 #just default to 0
	bestMean = 0
	for u in a:
		avg = np.mean(a[np.where(a[:,0] == u[0])][:, 1]) #calc mean reward for each action
		if bestMean < avg:
			bestMean = avg
			bestArm = u[0]
	return bestArm

plt.xlabel("Plays")
plt.ylabel("Avg Reward")
for i in range(1000):
	if i % 100 == 0:
		eps -= 0.01

	if random() > eps: #greedy arm selection
		choice = bestArm(av)
		thisAV = np.array([[choice, reward(arms[choice])]])
		av = np.concatenate((av, thisAV), axis=0)
	else: #random arm selection
		choice = np.where(arms == np.random.choice(arms))[0][0]
		thisAV = np.array([[choice, reward(arms[choice])]]) #choice, reward 
		av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
	#calculate the percentage the correct arm is chosen (you can plot this instead of reward)
	percCorrect = 100*(len(av[np.where(av[:,0] == np.argmax(arms))])/len(av))
	#calculate the mean reward
	runningMean = np.mean(av[:,1])
	plt.scatter(i, percCorrect, c='b', s=1)
plt.show()
