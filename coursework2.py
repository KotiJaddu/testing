#!/usr/bin/env python
# coding: utf-8

# ### Import Dependencies

# In[56]:


import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import DQN, ReplayBuffer, greedy_action, epsilon_greedy, update_target, loss

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

import gym
import matplotlib.pyplot as plt


# ### Hyperparameter Search
# Edit the `hyperparametersearch` variable for trialling different configurations of hyperparameters

# In[41]:


# Used to find the best configuration of hyperparameters
# Scoring is based on the total mean of the reward across all runs

NUM_RUNS = 2
NUM_EPISODES = 300
EPSILON_INIT = 1.0

hyperparameter_search_results = []

env = gym.make('CartPole-v1')

hyperparametersearch = {"OPTIMIZER": ["Adam", "SGD"],
                        "ACTIVATION_FUNCTION":["relu", "tanh", "sigmoid"],
                        "NETWORK": [[25], [50], [100], [25, 25], [50, 50], [100, 100]],
                        "BATCH_SIZE/REPLAY_BUFFER_SIZE" : [[32, 256], [32, 512], [64, 256], [64, 512]],
                        "EPSILON_FINAL" : [0.1, 0.01, 0.001, 0.0001],
                        "LEARNING_RATE" : [0.1, 0.01, 0.001, 0.0001],
                        "UPDATE_TARGET_FREQUENCY":[1, 5, 10, 25]
                       }

hyperparametersearch = {"OPTIMIZER": ["SGD"],
                        "ACTIVATION_FUNCTION":["sigmoid"],
                        "NETWORK": [[25]],
                        "BATCH_SIZE/REPLAY_BUFFER_SIZE" : [[32, 256]],
                        "EPSILON_FINAL" : [0.1],
                        "LEARNING_RATE" : [0.1],
                        "UPDATE_TARGET_FREQUENCY":[25]
                       }

params = []
total_steps = np.prod([len(trials) for trials in hyperparametersearch.values()])
step = 0

for OPTIMIZER in hyperparametersearch["OPTIMIZER"]:
    for ACTIVATION_FUNCTION in hyperparametersearch["ACTIVATION_FUNCTION"]:
        for NETWORK in hyperparametersearch["NETWORK"]:
            for BATCH_SIZE_REPLAY_BUFFER_SIZE in hyperparametersearch["BATCH_SIZE/REPLAY_BUFFER_SIZE"]:
                for EPSILON_FINAL in hyperparametersearch["EPSILON_FINAL"]:
                    for LEARNING_RATE in hyperparametersearch["LEARNING_RATE"]:
                        for UPDATE_TARGET_FREQUENCY in hyperparametersearch["UPDATE_TARGET_FREQUENCY"]:
                            run_results = []

                            current_parameters = {"OPTIMIZER": OPTIMIZER, "ACTIVATION_FUNCTION": ACTIVATION_FUNCTION, "NETWORK": NETWORK, "BATCH_SIZE/REPLAY_BUFFER_SIZE" : BATCH_SIZE_REPLAY_BUFFER_SIZE, "EPSILON_FINAL" : EPSILON_FINAL, "LEARNING_RATE" : LEARNING_RATE, "UPDATE_TARGET_FREQUENCY":UPDATE_TARGET_FREQUENCY}
                            params.append(str(current_parameters))
                            print(f"\nStep {step+1}/{total_steps}: {current_parameters}\n")
                            step += 1
                            for run in range(NUM_RUNS): # Execute each run
                                
                                print(f"run {run+1} of {NUM_RUNS}")
                                policy_net = DQN([4] + NETWORK + [2], ACTIVATION_FUNCTION)
                                target_net = DQN([4] + NETWORK + [2], ACTIVATION_FUNCTION)
                                update_target(target_net, policy_net)
                                target_net.eval()
                                
                                if OPTIMIZER == "Adam": # Either Adam or SGD
                                    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
                                else:
                                    optimizer = optim.SGD(policy_net.parameters(), lr=LEARNING_RATE)
                                memory = ReplayBuffer(BATCH_SIZE_REPLAY_BUFFER_SIZE[1])

                                steps_done = 0

                                episode_durations = []

                                for i_episode in range(NUM_EPISODES): # Run each episode
                                    
                                    if (i_episode+1) % 50 == 0:
                                        print("episode ", i_episode+1, "/", NUM_EPISODES)

                                    observation, info = env.reset()
                                    state = torch.tensor(observation).float()

                                    done = False
                                    terminated = False
                                    t = 0
                                    EPSILON_DECAY = (EPSILON_FINAL / EPSILON_INIT) ** (1 / (NUM_EPISODES - 50))
                                    epsilon = max(EPSILON_INIT * (EPSILON_DECAY ** i_episode), EPSILON_FINAL)
                                    while not (done or terminated):
                                        action = epsilon_greedy(epsilon, policy_net, state)
                                        
                                        observation, reward, done, terminated, info = env.step(action)
                                        reward = torch.tensor([reward])
                                        action = torch.tensor([action])
                                        next_state = torch.tensor(observation).reshape(-1).float()

                                        memory.push([state, action, next_state, reward, torch.tensor([done])])

                                        # Move to the next state
                                        state = next_state

                                        # Perform the optimization on the policy network
                                        if not len(memory.buffer) < BATCH_SIZE_REPLAY_BUFFER_SIZE[0]:
                                            transitions = memory.sample(BATCH_SIZE_REPLAY_BUFFER_SIZE[0])
                                            state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                                            # Compute loss
                                            mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                                            # Optimize the model
                                            optimizer.zero_grad()
                                            mse_loss.backward()
                                            optimizer.step()
                                        
                                        if done or terminated:
                                            episode_durations.append(t + 1)
                                        t += 1
                                    # Update the target network, copying all weights and biases in DQN
                                    if i_episode % UPDATE_TARGET_FREQUENCY == 0: 
                                        update_target(target_net, policy_net)
                                run_results.append(episode_durations)
                            hyperparameter_search_results.append(run_results.copy())
print('Completed Hyperparameter Search')


plt.plot(torch.arange(NUM_EPISODES) + 1, [100] * NUM_EPISODES, linestyle='dashed')

# Find best configuration of hyperparameters
best_score = -1
best_i = -1
for i in range(len(hyperparameter_search_results)):
    optimised_results = torch.tensor(hyperparameter_search_results[i])
    optimised_results_means = optimised_results.float().mean(0)
    score = np.mean(optimised_results_means.cpu().detach().numpy())
    if score > best_score:
        best_score = score
        best_i = i

# Plot the performances of configurations of hyperparameters
for i in range(len(hyperparameter_search_results)):
    optimised_results = torch.tensor(hyperparameter_search_results[i])
    optimised_results_means = optimised_results.float().mean(0)
    score = np.mean(optimised_results_means.cpu().detach().numpy())
    if i != best_i:
        plt.plot(torch.arange(NUM_EPISODES) + 1, optimised_results_means)
    else:
        plt.plot(torch.arange(NUM_EPISODES) + 1, optimised_results_means, linewidth=3.0, label=params[best_i])
print("Best Hyperparameters: " + params[best_i])

plt.title("Exploring Different Hyperparameters")
plt.ylabel("Return")
plt.xlabel("Episode Count")
plt.xlim([0, NUM_EPISODES])
plt.show()


#  ### Optimal Hyperparameters for DQN
# 
# 
# 

# In[57]:


NUM_RUNS = 2
NUM_EPISODES = 300

NETWORK = [4, 50, 50, 2]
ACTIVATION_FUNCTION = "relu"
OPTIMIZER = "Adam"

BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 512

EPSILON_FINAL = 0.001 
EPSILON_INIT = 1.0
EPSILON_DECAY = (EPSILON_FINAL / EPSILON_INIT) ** (1 / (NUM_EPISODES - 50))

LEARNING_RATE = 0.0004

NUM_EPISODES_PER_TARGET_UPDATE = 1


# ### Trainer for DQN and DDQN

# In[58]:


def train(NUM_RUNS=2, NUM_EPISODES=300, NETWORK=[4, 50, 50, 2], ACTIVATION_FUNCTION="relu", OPTIMIZER="Adam",
          BATCH_SIZE=64, REPLAY_BUFFER_SIZE=512, EPSILON_FINAL=0.001, LEARNING_RATE=0.0004,
          NUM_EPISODES_PER_TARGET_UPDATE=1, DDQN=False):
    """Trains the DQN or DDQN
    
    Args:
        NUM_RUNS: the number of replications to execute
        NUM_EPISODES: the number of episodes to execute per run
        NETWORK: the layout of the network including the input and output layer
        ACTIVATION_FUNCTION: the activation function applied after each layer in the forward pass
        OPTIMIZER: the optimizer to use for backpropagation
        BATCH_SIZE: how many samples should be used to train the network in each episode
        REPLAY_BUFFER_SIZE: the size of the buffer in the replay feature
        EPSILON_FINAL: the final exploration rate
        LEARNING_RATE: the learning rate for the optimizer
        NUM_EPISODES_PER_TARGET_UPDATE: the number of episodes between each time the target network is updated
        DDQN: if using the DDQN architecture
    
    Returns:
        runs_results: 2d array containing the reward collected at each episode for each run
        policy_net: the policy network trained on the final replication
    """
    runs_results = []
    env = gym.make('CartPole-v1')
    for run in range(NUM_RUNS):
        print(f"Starting run {run+1} of {NUM_RUNS}")
        policy_net = DQN(NETWORK)
        target_net = DQN(NETWORK)
        update_target(target_net, policy_net)
        target_net.eval()
        if OPTIMIZER=="Adam":
            optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = optim.SGD(policy_net.parameters(), lr=LEARNING_RATE)
        memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        steps_done = 0
        episode_durations = []

        for i_episode in range(NUM_EPISODES):

            if (i_episode+1) % 50 == 0:
                print("episode ", i_episode+1, "/", NUM_EPISODES)

            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            
            EPSILON_INIT = 1.0
            EPSILON_DECAY = (EPSILON_FINAL / EPSILON_INIT) ** (1 / (NUM_EPISODES - 50))
            EPSILON = max(EPSILON_INIT * (EPSILON_DECAY ** i_episode), EPSILON_FINAL)
            while not (done or terminated):
                action = epsilon_greedy(EPSILON, policy_net, state)

                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform the optimization on the policy network
                if not len(memory.buffer) < BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones, DDQN=DDQN)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()

                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1
            # Update the target network, copying all weights and biases in DQN
            if i_episode % NUM_EPISODES_PER_TARGET_UPDATE == 0: 
                update_target(target_net, policy_net)
        runs_results.append(episode_durations)
    print('Complete')
    return runs_results, policy_net
    


# ### Train DQN with Optimal Hyperparameters
# 

# In[ ]:


runs_results, policy_net = train()


# ### Visualise Learning Curve of Optimal DQN

# In[ ]:


# Calculate the mean and standard deviation values for each episode
results = torch.tensor(runs_results)
means = results.float().mean(0)
stds = results.float().std(0)

print(f"score: {np.mean(means.cpu().detach().numpy())}")

# Plot the calculated values for each episode
plt.plot(torch.arange(NUM_EPISODES) + 1, [100] * NUM_EPISODES, linestyle='dashed', color='b')
plt.plot(torch.arange(NUM_EPISODES) + 1, means, color='r')
plt.fill_between(np.arange(NUM_EPISODES), means, means-stds, alpha=0.4, color='r')
plt.fill_between(np.arange(NUM_EPISODES), means, means+stds, alpha=0.4, color='r')

# Format the chart
plt.ylabel("Return")
plt.xlabel("Episode")
plt.title("Learning Curve of the Optimised DQN over 10 Runs")
plt.xlim([0, NUM_EPISODES])

plt.show()


# ### Visualise Slices of the Greedy Policy Action and Q Function

# In[50]:


# Visualising the greedy Q-values for a stationary cart in the middle of the track
# 2D plot showing policy as a function of pole angle and angular velocity (omega)

# This plots the policy and Q values according to the network currently
# stored in the variable "policy_net"

q = True # whether q values or greedy policy is visualised

angle_range = .2095
omega_range = 2.1

cart_velocity = [[0, 0.5], [1, 2]]
angle_samples = 100
omega_samples = 100

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(8, 6)
fig.suptitle(("Greedy Q-Values" if q else "Greedy Policy") + f" when Cart Position is 0", fontsize=16)

for a in range(len(cart_velocity)):
    for b in range(len(cart_velocity[a])):
        # Iterate thorugh each cart velocity value
        angles = torch.linspace(angle_range, -angle_range, angle_samples)
        omegas = torch.linspace(-omega_range, omega_range, omega_samples)
        greedy_q_array = torch.zeros((angle_samples, omega_samples))
        policy_array = torch.zeros((angle_samples, omega_samples))
        for i, angle in enumerate(angles):
            for j, omega in enumerate(omegas):
                # Fill in values for contour plotting
                state = torch.tensor([0., cart_velocity[a][b], angle, omega])
                with torch.no_grad():
                    q_vals = policy_net(state)
                    greedy_action = q_vals.argmax()
                    greedy_q_array[i, j] = q_vals[greedy_action]
                    policy_array[i, j] = greedy_action
    
        if q:
            contour = axs[a, b].contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=100)
            plt.colorbar(contour, ax=axs[a,b]).set_label("Greedy Q-Value") # Add colourbar for each subplot
        else:
            contour = axs[a, b].contourf(angles, omegas, policy_array.T, cmap='cividis')
            axs[a, b].legend([contour.legend_elements()[0][0], contour.legend_elements()[0][-1]], ["Left", "Right"] )
        axs[a, b].set_title(f"Cart Velocity is {cart_velocity[a][b]}") # Set titles for each subplot

for ax in axs.flat:
    ax.set(xlabel='Pole Angle', ylabel='Pole Angular Velocity') # Set labels for the axes of each subplot

plt.tight_layout()
plt.show()


# ### Train DDQN with Optimal Hyperparameters of DQN

# In[55]:


runs_results_ddqn, _ = train(DDQN=True)


# ### Visualise DQN and DDQN Learning Curves

# In[53]:


# Calculate the mean and standard deviation values for each episode for dqn and ddqn runs
optimised_results = torch.tensor(runs_results)
optimised_results_means = optimised_results.float().mean(0)
optimised_results_stds = optimised_results.float().std(0)

optimised_results_ddqn = torch.tensor(runs_results_ddqn)
optimised_results_means_ddqn = optimised_results_ddqn.float().mean(0)
optimised_results_stds_ddqn = optimised_results_ddqn.float().std(0)

# Calculate scores for the mean and stds of dqn and ddqn runs
score_dqn = str(np.format_float_positional(np.mean(optimised_results_means.cpu().detach().numpy()), precision=5, unique=False, fractional=False, trim='k'))
score_ddqn = str(np.format_float_positional(np.mean(optimised_results_means_ddqn.cpu().detach().numpy()), precision=5, unique=False, fractional=False, trim='k'))
score_std_dqn = str(np.format_float_positional(np.mean(optimised_results_stds.cpu().detach().numpy()), precision=5, unique=False, fractional=False, trim='k'))
score_std_ddqn = str(np.format_float_positional(np.mean(optimised_results_stds_ddqn.cpu().detach().numpy()), precision=5, unique=False, fractional=False, trim='k'))

# Plot the calculated values for each episode for dqn and ddqn runs
plt.plot(torch.arange(NUM_EPISODES) + 1, [100] * NUM_EPISODES, linestyle='dashed')
plt.plot(torch.arange(NUM_EPISODES) + 1, optimised_results_means, color='r', label=f"DQN; Mean Score: {score_dqn}; Std Score: {score_std_dqn}")
plt.fill_between(np.arange(NUM_EPISODES) + 1, optimised_results_means, optimised_results_means-optimised_results_stds, alpha=0.2, color='r')
plt.fill_between(np.arange(NUM_EPISODES) + 1, optimised_results_means, optimised_results_means+optimised_results_stds, alpha=0.2, color='r')
plt.plot(torch.arange(NUM_EPISODES) + 1, optimised_results_means_ddqn, color='g', label=f"DDQN; Mean Score: {score_ddqn}; Std Score: {score_std_ddqn}")
plt.fill_between(np.arange(NUM_EPISODES) + 1, optimised_results_means_ddqn, optimised_results_means_ddqn-optimised_results_stds_ddqn, alpha=0.2, color='g')
plt.fill_between(np.arange(NUM_EPISODES) + 1, optimised_results_means_ddqn, optimised_results_means_ddqn+optimised_results_stds_ddqn, alpha=0.2, color='g')

# Format the chart
plt.ylabel("Return")
plt.xlabel("Episode")
plt.title("Optimised DQN vs DDQN Performance over 10 Runs")
plt.xlim([0, NUM_EPISODES])
plt.legend()
plt.show()


# In[ ]:




