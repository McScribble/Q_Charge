# Q_Charge

Repository containing the Q_Charge AI. this AI utilizes implements a Q learning algorithm for a robot to navigate its environment to its charging station.
Q learning is a reinforcement based learning algorithm, associated with Q values that determine the efficacy
of a given decision at a given state.

## Q Learning 

The Q learning algorithm is commonly utilized to optimize navigation, and is based on a table of Q values that
represent the optimal control decision for a given state. These Q values are calculated by rewarding the
algorithm as it completes desired objectives. The destination, or target parameter is associated with the
initial reward, and the algorithm will take random action until it is able to find a reward, at which state
it will begin calculating potential reward at each state leading to the found reward. In this way, it iteratively
builds up a reward at each state, stemming from the initial reward state. These rewards are calculated using
the Q equation described here: https://en.wikipedia.org/wiki/Q-learning

## The Environment

The robot's environment is a 2 dimensional grid, containing walls and a target objective, the charging pad.

The spaces in the grid are represented by the following symbols:

Available open space:

![Alt text](grass.png?raw=true "Title")

Robot:

![Alt text](Robot.png?raw=true "Title")

Charger:

![Alt text](charger.png?raw=true "Title")

Wall (obstacle):

![Alt text](wall.png?raw=true "Title")


The environment is the following maze:

![Alt text](grid.png?raw=true "Title")


## The Code

The full code for this algorithm is implemented in Q_Charge.py

### RobotEnvironment

This class contains all of the information about the grid maze, all decisions that are available to the robot, the drawing logic for the 2D representation, and updates to
the environment as the robot makes decisions.

### Cell

This class contains the data for a single cell in the grid

### Q Learning Loop

This portion of the code starts the Q learning process, and updates the Q table as the robot begins to make decisions in the environment. 
The Q table is displayed with total reward per action at every state at the 1st, 10th, and last iterations of the algorithm

### Reward Growth Curve

This curve shows how the robot has accumulated higher reward over the course of training in the environment, displaying reward vs episodes

### Q Robot Plays game

This shows the trained Q Robot playing the game on the 2D grid.
