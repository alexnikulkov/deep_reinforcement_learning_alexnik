[//]: # (Image References)

[actor_network]: images/actor_network.png "Actor Network"
[critic_network]: images/critic_network.png&s=50 "Critic Network"
[rewards]: images/rewards.png&s=50 "Rewards"

# Continuosus Control Report

## Learning Algorithm

The agent is trained using Deep Deterministic Policy Gradient (DDPG) method, which simultaneously learns an actor (a network which decides which actions to take) and a critic (a network which computes Q-values for every possible actions at every state). While only the actor netowrk is needed to choose the optimal actions after training is finished, the critic network is necessary for the actor to evaluate possible actions it could take during training.

Both actor and critic networks have target copies which are used to generate target values for critic's training. These copies are updated at every step by mixing 0.1% of the main networks' weights into the target networks.

Noise is applied to the actions suggested by the actor to encourage exploration. The noise is generated from an Orstein-Uhlenbeck process with gradually decreaing variance.

# Network Architectures and Hyperparameters

### Actor Network

The actor network takes a 33-dimensional state vector as a input and it outputs the 4-dimensional action vector. It is a fully-connected network with 2 hidden layers. The 1st hidden layer has 256 units and the 2nd hidden unit has 128 units. Each hidden layer uses a ReLU activation function and the output uses a tanh activation to map the actions to (-1;1) interval. The full actor network visualization (produced by torchviz) is shown below. 

![Actor Network][actor_network]

### Critic Network

The critic network takes a 33-dimensional state vector and a 4-dimensional action vector as inputs and returns a scalar (predicted Q-value). The architecture is similar to a fully-connected network with 2 hidden layers, with a slight modification that the action vector is inputted not at the input layer, but at the 1st hidden layer. This modification was suggested in the original DDPG paper and I hypothesized that it enhances the performance by blending the action vector with higher-level representation of the state. The 1st hidden layer has 256 units and the 2nd hidden unit has 128 units. Each hidden layer uses a ReLU activation function and the output layer uses an idenity activation function. The full critic network visualization (produced by torchviz) is shown below.

![Critic Network][critic_network]

### Hyperparameters

Parameter | Value
--- | ---
Number of agents simulated in parallel | 20
Replay buffer size | 100,000
Batch size | 128
Gamma (discount multiplier) | 0.99
Noise theta | 0.15
Noise sigma | 0.2
Noise sigma multiplier (per episode) | 0.99
Network weight decay coefficient | 1e-6
Learning rate multiplier (per episode) | 0.996
Actor hidden layer sizes | [256,128]
Actor initial learning rate | 1e-4
Critic hidden layer sizes | [256, 128]
Critic intitial learning rate | 1e-4

## Results

The agent learns very rapidly. It first reaches per-episode average reward of 30 after just 30 episodes and eventually it reaches average rewards of 36-38. The plot below shows how rewards change over time.

![Rewards Plot][rewards]

## Ideas for Improvement

The algorthm can be further improved by:
1. Implementing prioritized experience replay to focus learning on the most important experiences.
2. Changing the actor to produce not deterministic actions, but sotchastic actions which would naturally explore.