[//]: # (Image References)

[trained_agent]: images/trained_agents.gif "Trained Agents"

# Udacity Deep Reinforcement Learning Nanodegree Project 3: Collaboration and Competition

### Introduction

In this project I trained a pair of agents to play tennis collaboratively in a [Tennis Unity environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) 

![Trained Agents][trained_agents]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Instructions

1. Make sure you have Python 3.6 + Jupyter and some standard libraries (pandas, numpy) installed.

2. Install Unity.

3. Install [Unity MLAgents](https://github.com/Unity-Technologies/ml-agents).

4. Download the Tennis app, unzip (or decompress) the file, place it into the project folder and rename it to Tennis.app (for MacOS):
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

5. Run Tennis.ipynb in Jupyter to train the agents.
