[//]: # (Image References)

[rewards]: images/rewards.png "Rewards"

# Collaboration and Competition Project Report

## Learning Algorithm

The agents are trained using  Multi-Agent Deep Deterministic Policy Gradient (MADDPG) method, which simultaneously learns an actor (a network which decides which actions to take) and a critic (a network which computes Q-values for every possible actions at every state) for each agent. While only the actor netowrk is needed to choose the optimal actions during inference/testing, the critic network is necessary for the actor to evaluate possible actions it could take during training. A distinct feature of MADDPG is that it uses a "global" critic for each agent, which takes as inputs the states and actions of all agents, not just the agent which is being trained. This helps reduce the negative impact of non-stationarity of the environment from the point of view of each actor due to changing policies of other agents.

Both actor and critic networks have target copies which are used to generate target values for critic's training. These copies are updated at every step by mixing 3% of the main networks' weights into the target networks.

Noise is applied to the actions suggested by the actor to encourage exploration. The noise is generated from an Orstein-Uhlenbeck process with gradually decreaing variance. During the first 300 episodes the agents take random actions (uniform on [-1;1]) to provide a sufficiently diverse set of experiences to fill the replay buffer.

# Network Architectures and Hyperparameters

### Actor Network

The actor network takes a 8-dimensional state vector as a input and it outputs the 2-dimensional action vector. It is a fully-connected network with 2 hidden layers. Each of the hidden units has 64 neuronsm and uses a ReLU activation function, while the output layer uses a tanh activation to map the actions to (-1;1) interval.

### Critic Network

The critic network takes a pair of 8-dimensional state vectors and a pair of 2-dimensional action vectors as inputs (24 inoputs in total) and returns a scalar (predicted Q-value). The architecture is similar to a fully-connected network with 2 hidden layers, with a slight modification that the action vector is inputted not at the input layer, but at the 1st hidden layer. This modification was suggested in the original DDPG paper and I hypothesized that it enhances the performance by blending the action vector with higher-level representation of the state. Each of the hidden layers has 64 units and uses a ReLU activation function, while the output layer uses an idenity activation function.

### Hyperparameters

Parameter | Value
--- | ---
Replay buffer size | 1,000,000
Batch size | 128
Gamma (discount multiplier) | 0.99
Noise theta | 0.15
Noise sigma | 0.3
Noise sigma multiplier (per episode) | 0.997
Network weight decay coefficient | 0
Learning rate multiplier (per episode) | 0.9986
Actor hidden layer sizes | [64,64]
Actor initial learning rate | 1e-3
Critic hidden layer sizes | [64, 64]
Critic intitial learning rate | 3e-3

## Results

The agents initially perform note very well, but after 1500 episodes the performance dramatically improves and they reach the running average of 0.5 points per episode after 1673 episodes, with rewards during some epsodes reaching 2.5 (agents hit the ball over the net 25 times) The plot below shows how rewards change over time.

![Rewards Plot][rewards]

## Ideas for Improvement

The algorthm can be further improved by:
1. Incentivizing the agents to seek exploration, instead of forcing it onto them as noise.
2. Allowing the agents to communicate with each other.