# Q-Charge

# this script implements a Q learning algorithm for a robot to navigate its environment to its charging station.
# Q learning is a reinforcement based learning algorithm, associated with Q values that determine the efficacy
# of a given decision at a given state.
#
# The Q learning algorithm is commonly utilized to optimize navigation, and is based on a table of Q values that
# represent the optimal control decision for a given state. These Q values are calculated by rewarding the
# algorithm as it completes desired objectives. The destination, or target parameter is associated with the
# initial reward, and the algorithm will take random action until it is able to find a reward, at which state
# it will begin calculating potential reward at each state leading to the found reward. In this way, it iteratively
# builds up a reward at each state, stemming from the initial reward state. These rewards are calculated using
# the Q equation described here: https://en.wikipedia.org/wiki/Q-learning


# please ensure that you have the following modules installed to run this code,or you activate the virtual environment
# in the venv folder
# numpy, matplotlib, pillow
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from PIL import Image


# Classes developed for code


# define the robots environment
class RobotEnvironment:

    def __init__(self):

        # defines the map of cells
        self.cells = {
            0: Cell('Ground', 1, 1, 0, 'grass.png'),
            1: Cell('Ground', 1, 2, 0, 'grass.png'),
            2: Cell('Charger', 1, 3, 100, 'charger.png'),
            3: Cell('Ground', 1, 4, 0, 'grass.png'),
            4: Cell('Ground', 1, 5, 0, 'grass.png'),
            5: Cell('Ground', 2, 1, 0, 'grass.png'),
            6: Cell('Ground', 2, 2, 0, 'grass.png'),
            7: Cell('Wall', 2, 3, 0, 'wall.png'),
            8: Cell('Wall', 2, 4, 0, 'wall.png'),
            9: Cell('Ground', 2, 5, 0, 'grass.png'),
            10: Cell('Ground', 3, 1, 0, 'grass.png'),
            11: Cell('Ground', 3, 2, 0, 'grass.png'),
            12: Cell('Ground', 3, 3, 0, 'grass.png'),
            13: Cell('Ground', 3, 4, 0, 'grass.png'),
            14: Cell('Ground', 3, 5, 0, 'grass.png'),
            15: Cell('Ground', 4, 1, 0, 'grass.png'),
            16: Cell('Wall', 4, 2, 0, 'wall.png'),
            17: Cell('Wall', 4, 3, 0, 'wall.png'),
            18: Cell('Ground', 4, 4, 0, 'grass.png'),
            19: Cell('Ground', 5, 5, 0, 'grass.png'),
            20: Cell('Ground', 5, 1, 0, 'grass.png'),
            21: Cell('Ground', 5, 2, 0, 'grass.png'),
            22: Cell('Ground', 5, 3, 0, 'grass.png'),
            23: Cell('Ground', 5, 4, 0, 'grass.png'),
            24: Cell('Ground', 5, 5, 0, 'grass.png'),
        }
        self.active_cell = 22  # start cell
        self.states = 25  # number of possible states
        self.actions = 4  # number of possible actions
        self.directions = ['left', 'up', 'right', 'down']  # directions that are available

    # defines directional movement
    def step(self, direction, update=False):

        # inititalize current position
        cell = self.cells[self.active_cell]
        row = cell.row
        col = cell.col
        nextindex = [row, col]

        # put new (row,col) position in nextindex
        if direction == 'left' and col > 1:

            nextindex = [row, col - 1]

        elif direction == 'up' and row > 1:

            nextindex = [row - 1, col]

        elif direction == 'right' and col < 5:

            nextindex = [row, col + 1]

        elif direction == 'down' and row < 5:

            nextindex = [row + 1, col]

        # find single index of the next cell
        nextcellind = self.getnextcell(nextindex)

        # return the next cell
        nextcell = self.cells[nextcellind]

        # check if the next position is the target charger
        if nextcell.name == 'Charger':
            self.active_cell = self.getnextcell(nextindex)

            # update GUI
            if update:
                self.draw()

            return self.active_cell, nextcell.reward, True

        # if it does not hit a wall, move to next state
        elif nextcell.name != 'Wall':
            self.active_cell = self.getnextcell(nextindex)

            # update GUI
            if update:
                self.draw()

            return self.active_cell, nextcell.reward, False

        # if it hit wall or boundary, do not move
        else:

            return self.active_cell, cell.reward, False

    # get the next cell number from its row col index
    def getnextcell(self, index):

        row = index[0]
        col = index[1]

        return ((row - 1) * 5 + col) - 1

    # draw the grid map
    def draw(self):

        # utilize PIL to draw
        cells = self.cells
        active_cell = self.active_cell

        if 0 == active_cell:
            array = np.array([np.asarray(Image.open('Robot.png').convert('RGB'))])
        else:
            array = np.array([np.asarray(Image.open(cells[1].disp).convert('RGB'))])

        for cell in cells:

            if cell == 0:
                continue

            if cell == active_cell:
                array2 = np.array([np.asarray(Image.open('Robot.png').convert('RGB'))])
            else:
                array2 = np.array([np.asarray(Image.open(cells[cell].disp).convert('RGB'))])

            array = np.vstack((array, array2))

        ncols = 5
        nindex, height, width, intensity = array.shape
        nrows = nindex // ncols
        assert nindex == nrows * ncols
        endgrid = (array.reshape(nrows, ncols, height, width, intensity)
                   .swapaxes(1, 2)
                   .reshape(height * nrows, width * ncols, intensity))

        self.ax.imshow(endgrid)
        plt.ion()
        plt.show()
        plt.pause(0.01)

    # reset the environment
    def reset(self, ax=''):

        self.ax = ax
        self.active_cell = 22


# encapsulates the information that a single map cell can hold
class Cell:
    def __init__(self, name, row, col, reward, disp):
        self.name = name
        self.row = row
        self.col = col
        self.reward = reward
        self.disp = disp


# Q Learning Loop


print('Running Q Loop')
# initialize environment
env = RobotEnvironment()

# initialize q table (make everything zeros)
qtable = np.zeros((env.states, env.actions))
print(qtable)

# set the learning characteristics
total_episodes = 100  # number of episodes that the learning algorithm will carry out
learning_rate = 0.7  # the rate at which the algorithm updates the q-table
discount_rate = 0.9  # the rate of discount for favorable moves
max_steps = 30  # max n=umber of steps the robot will take before termination of an episode
greedy_policy = .75  # defines the likelihood that the robot will take the best option on the q-table over a random step
reward_curve = np.zeros(total_episodes)  # initialize the reward curve to zero

# start learning process
for episode in range(total_episodes):

    # set up initial environment for episode and state
    env.reset()
    state = env.active_cell
    done = False
    step = 0
    total_reward = np.zeros(max_steps)

    # begin making moves
    for step in range(max_steps):

        # check if greedy policy initiates
        greedy_prob = random.uniform(0, 1)

        if greedy_prob < greedy_policy:
            action = random.randint(0, 3)
        else:
            action = np.argmax(qtable[state, :])

        # map action number to direction
        direction = env.directions[action]

        # make move
        new_state, reward, done = env.step(direction)

        # estimate 1 value with the q equations
        qtable[state, action] = (qtable[state, action] + learning_rate *
                                 (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]))

        # add to the reward for this step
        total_reward[step] += reward + qtable[state, action]

        state = new_state

        if done:
            break

    if episode == 0 or episode == 9 or episode == total_episodes - 1:
        print('\nQ Table at episode ' + str(episode + 1) + ':\n')
        print(qtable)
        input('\nPress any key to continue...\n')

    reward_curve[episode] = np.average(total_reward)

# this curve shows the reward growth of the AI over each iteration
fig, ax = plt.subplots()  # create an axis for the reward curve
ax.plot(list(range(0, total_episodes)), reward_curve)
ax.set_title('Reward Growth Curve')
ax.set_ylabel('Reward')
ax.set_xlabel('Episodes')
plt.ion()
plt.show()
plt.pause(0.01)

### Q-Robot Plays Game (visualized)

fig2, ax = plt.subplots()
env.reset(ax)

print('\n\nThe trained Q AI will now play the game\n')
input('\nPress any key to start the game')

env.draw()

done = False
move = 0

while not done:
    state = env.active_cell
    step = 0
    move += 1
    print('\n\n\nMove : ' + str(move))
    print("\n\nQ Choices at current state:\n")
    print(qtable[state, :])

    action = np.argmax(qtable[state, :])
    direction = env.directions[action]
    print('\n\nNext Action:\n')
    print(str(action) + ' : ' + direction)

    input('\nPress any key to make action')

    new_state, reward, done = env.step(direction, True)
    env.draw()

input('\nPress any key to exit')
