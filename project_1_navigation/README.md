[//]: # (Image References)

[trained_agent]: images/trained_agent.gif "Trained Agent"
[plot_rewards]: images/plot_rewards.png "Rewards Plot"

# Udacity Deel Reinforcement Learning Nanodegree Project 1: Navigation

## Introduction

In this project I trained an agent to navigate a square world and collect bananas along the way.

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

## Methods

An agent was trained using Deep Q-Learning (DQL) with some modifications listed below. These modifications have been shown in the literature to improve performance, but I have not performed and ablation study to validate the need for each modification. The following modifications were applied:
1. [Double Q-Learning](https://arxiv.org/abs/1509.06461). A second (target) network was used to evaluate the action value of the best action in each state. This helped reduce the overestimation bias inherent to Q-Learning.
2. [Dueling network architecture](https://arxiv.org/abs/1511.06581). Each action value was decomposed into a sum of state value (common for all actions) and an advantage, which has mean of 0 across all possible actions in a state. This allowed me to use the fact that action values in the same state are often strongly correlated because an immediate next action often has less impact on the action value than the state from which the action is taken, thus re-using the updated state value across all actions in the state.
3. [Prioritized experience replay](https://arxiv.org/abs/1511.05952). Instead of uniformaly sampling from all saved experiences, I sampled from experiences which had a larger discrepancy between the target and the current estimate more often. This modification focused model training on more informative experiences and accelerated learning.
4. State consists of multiple frames. My agent supports using multiple previous frames and actions. I experimented with this ieda, but couldn't get it to outperform the baseline, so it's included in the code, but wasn't enabled for the results which I'm presenting.

The Q network was fully connected (except for the connections to the output layer, which had "dueling" architecture) with 2 hidden layers with dimensions 64, 32. The learning rate started at 5e-4 and was exponentially decayed with a multiplier of 0.997 per episode, with a lower bound of 1e-6. The main Q network was updated every 4 steps with a batch size of 128. During each update 3 gradient descent steps were taken by the Adam optimizer with default parameters.

## Results

I was able to train the agent to exceed a target average episode reward (average over 100 consecutive episodes) of 13.0 in 467 episodes. The plot below shows smoothed reward function as a function of episode number. The agent takes several hundred episodes to explore the space, followed by rapid learning process and eventual saturation of the reward function. An example of an agent following a learned policy is shown in the Introduction.

![Rewards Plot][plot_rewards]

## Future Work

My result is by no means perfect, so multiple improvements can be made:

1. Accelerate runtime. It took me around 30 seconds to simulate a single episode even without model training. During execution my CPU load was very light (<10%), so there might be an opportunity to make the code run faster by allowing Unity to make full use of the CPU.
2. Hyperparameter tuning. As is typical in Deep Learning, I found my model to be very sensitive to hyperparameters. Long runtimes prevented me from performing an exhaustive hyperparameter tuning, but it could be done if model runtime is accelerated.
3. Blocking net-zero action loops.The trained agent sometimes showed strange sequences of net-zero actions. For example, turning lert-righ-left-right-... for 10+ time steps. If this can be solved, I believe the agent performance could be improved significantly. If I had more time, I would have investigated these instances in detail by analyzing the Q values based on which the agent was getting stuck in these loops. I'm not sure what could be a solution, but it could be inspired by the results of this analysis. One interesting option could be to include previous actions as part of the state space.