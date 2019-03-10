[//]: # (Image References)

[trained_agent]: images/trained_agent.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree Project 1: Navigation

## Introduction

In this project I trained an agent to navigate a square world and collect bananas along the way. An example of trained agent behavior can be seen below.

![Trained Agent][trained_agent]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Make sure you have Python 3.6 + Jupyter and some standard libraries (pandas, numpy) installed.

2. Install Unity.

3. Install [Unity MLAgents](https://github.com/Unity-Technologies/ml-agents).

4. Run the Report.ipynb notebook in Jupyter.