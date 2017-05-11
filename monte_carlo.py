import numpy as np
import random
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from monte_carlo_utils import initStateSpace, initStateActions, initSAcount, calcReward, updateQtable, qsv, getRLstate
from blackjack import randomCard, useable_ace, totalValue, add_card, eval_dealer, new_deck, play, initGame

epochs = 5000000
testing_set = 1000
epsilon = 0.1

state_space = initStateSpace()
av_table = initStateActions(state_space)
av_count = initSAcount(av_table)

# training
for i in range(epochs):
    #initialize new game; observe current state
    state = initGame()
    player_hand, dealer_hand, status = state
    #if player's total is less than 11, increase total by adding another card
    #we do this because whenever the player's total is less than 11, you always hit no matter what
    #so we don't want to waste compute cycles on that subset of the state space
    while player_hand[0] < 11:
        player_hand = add_card(player_hand, randomCard())
        state = (player_hand, dealer_hand, status)
    rl_state = getRLstate(state) #convert to compressed version of state
    
    #setup dictionary to temporarily hold the current episode's state-actions
    returns = {} #state, action, return
    while(state[2] == 1): #while in current episode
        #epsilon greedy action selection
        act_probs = qsv(rl_state, av_table)
        if (random.random() < epsilon):
            action = random.randint(0,1)
        else:
            action = np.argmax(act_probs)#select an action
        sa = ((rl_state, action))
        returns[sa] = 0 #add a-v pair to returns list, default value to 0
        av_count[sa] += 1 #increment counter for avg calc
        state = play(state, action) #make a play, observe new state
        rl_state = getRLstate(state)
    #after an episode is complete, assign rewards to all the state-actions that took place in the episode
    for key in returns: 
        returns[key] = calcReward(state[2])
    av_table = updateQtable(av_table, av_count, returns)

# testing
total = 0
for i in range(1000):
    state = initGame()
    player_hand, dealer_hand, status = state
    while player_hand[0] < 11:
        player_hand = add_card(player_hand, randomCard())
        state = (player_hand, dealer_hand, status)
    rl_state = getRLstate(state) #convert to compressed version of state
    
    returns = {} #state, action, return
    while(state[2] == 1):
        act_probs = qsv(rl_state, av_table)
        action = np.argmax(act_probs)
        state = play(state, action)
    total += calcReward(state[2])

print(total)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d', )

# ax.set_xlabel('Dealer card')
# ax.set_ylabel('Player sum')
# ax.set_zlabel('State-Value')

# x,y,z = [],[],[]
# for key in state_space:
#     if (not key[1] and key[0] > 11 and key[2] < 21):
#         y.append(key[0])
#         x.append(key[2])
#         state_value = max([av_table[(key, 0)], av_table[(key, 1)]])
#         z.append(state_value)
# ax.azim = 230
# ax.plot_trisurf(x,y,z, linewidth=.02, cmap=cm.jet)

# plt.show()
print("Done")