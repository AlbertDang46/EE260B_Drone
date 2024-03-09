# from __future__ import annotations

# import random

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import torch
# import torch.nn as nn
# from torch.distributions.normal import Normal

# import gymnasium as gym


# plt.rcParams["figure.figsize"] = (10, 5)

# env = gym.make("InvertedPendulum-v4", render_mode="human")
# observation, info = env.reset(seed=42)

# for _ in range(1000):
#    action = env.action_space.sample()
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()

# env.close()

import torch
import gymnasium as gym

gym_env_name = 'CartPole-v1'

train_env = gym.make(gym_env_name)

max_steps = 50_000
step = 0
lr = 0.005
gamma = 0.9999

nn = torch.nn.Sequential(
   torch.nn.Linear(4, 64),
   torch.nn.ReLU(),
   torch.nn.Linear(64, train_env.action_space.n),
   torch.nn.Softmax(dim=-1)
)
optim = torch.optim.Adam(nn.parameters(), lr=lr)

while step < max_steps:
   init_obs, info = train_env.reset()
   obs = torch.tensor(init_obs, dtype=torch.float)    
   done = False
   Actions, States, Rewards = [], [], []

   while not done:
      probs = nn(obs)
      dist = torch.distributions.Categorical(probs=probs)        
      action = dist.sample().item()
      next_obs, rew, terminated, truncated, info = train_env.step(action)
      done = terminated or truncated
      
      Actions.append(torch.tensor(action, dtype=torch.int))
      States.append(obs)
      Rewards.append(rew)

      obs = torch.tensor(next_obs, dtype=torch.float)
      step += 1
        
   DiscountedReturns = []
   for t in range(len(Rewards)):
      G = 0.0
      for k, r in enumerate(Rewards[t:]):
         G += (gamma**k) * r
      DiscountedReturns.append(G)
   
   for k, r in enumerate(zip(States, Actions, DiscountedReturns)):
      State, Action, G = r

      probs = nn(State)
      dist = torch.distributions.Categorical(probs=probs)    
      log_prob = dist.log_prob(Action)
      
      loss = -G * log_prob
      
      optim.zero_grad()
      loss.backward()
      optim.step()

train_env.close()

# Test trained model

test_env = gym.make(gym_env_name, render_mode='human')

for _ in range(5):
   Rewards = []
   
   init_obs, info = test_env.reset()
   obs = torch.tensor(init_obs, dtype=torch.float)   
   done = False
   
   while not done:
      probs = nn(obs)
      dist = torch.distributions.Categorical(probs=probs)        
      action = dist.sample().item()
      
      next_obs, rew, terminated, truncated, info = test_env.step(action)
      done = terminated or truncated

      obs = torch.tensor(next_obs, dtype=torch.float)
      Rewards.append(rew)
   
   print(f'Reward: {sum(Rewards)}')

test_env.close()