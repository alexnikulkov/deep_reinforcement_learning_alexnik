[//]: # (Image References)

[trained_agent]: images/trained_agent.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree Project 2: Continuosus Control

## Introduction

In this project I trained an agent to control a double-jointed arm to reach and follow a moving target. It is based on a [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) Unity environment.

![Trained Agent][trained_agent]

A reward of +0.1 is provided for every time step when the tip of the arm is inside the target sphere.  Thus, the goal of the agent is to reach the target sphere as fast as possible and to remain inside of it for as long as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints (two torque dimensions per joint). Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

20 identical and independent agents are synchronously simulated in parallel to accelerate learning. All agents use the same copy of the network to take actions and at every step the experiences of all 20 agensts are added to the replay buffer. The networks are updated at each step by sampling from the replay buffer.

## Instructions

1. Make sure you have Python 3.6 + Jupyter and some standard libraries (pandas, numpy) installed.

2. Install Unity.

3. Install [Unity MLAgents](https://github.com/Unity-Technologies/ml-agents).

4. Download the Reacher app, unzip (or decompress) the file, place it into the project folder and rename it to Reacher_20.app:
	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

5. Run the Continuous_Control.ipynb notebook in Jupyter.