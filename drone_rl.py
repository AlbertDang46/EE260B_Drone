import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Drone_Environment:
    """PX4 + Gazebo drone environment for reinforcement learning"""

    def __init__(self):
        self.setup()

    # Wait for drone services to come online and setup telemetry
    def setup(self):
        print("setup")

    # De-arm and re-arm drone and reset drone position
    def reset(self):
        print("reset")

    #
    def step(self):
        print("step")

    # De-arm drone and reset drone position
    def shutdown(self):
        print("shutdown")
       

class Drone_Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
        of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
        of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(input.float())

        action_means = self.policy_mean_net(shared_features)
        
        # Keep standard deviation positive even if stddev_net outputs a negative value
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


class Drone_REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Drone_Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        # Get predicted mean and standard deviation from policy network
        state_tensor = torch.tensor(state)
        action_means, action_stddevs = self.net(state_tensor)

        # Create a normal distribution from the predicted mean and standard deviation and sample an action
        norm_distrib = Normal(action_means + self.eps, action_stddevs + self.eps)
        action = norm_distrib.sample()

        prob = norm_distrib.log_prob(action)
        self.probs.append(prob)

        action = action.tolist()
        return action

    def update(self):
        """Updates the policy network's weights."""
        running_G = 0
        Gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_G = R + self.gamma * running_G
            Gs.insert(0, running_G)

        Gs_tensor = torch.tensor(Gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for G, log_prob in zip(Gs_tensor, self.probs):
            loss += (-1) * G * log_prob.mean()

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


def print_neural_net_parameters(neural_net):
    for param in neural_net.parameters():
        print(type(param), param.size(), param)


drone_env = Drone_Environment()
total_num_episodes = 5000

# Drone State: x,y,z global coordinates and x,y,z linear velocities
obs_space_dims = 6
# Drone Action: x,y,z linear velocities
action_space_dims = 3
drone_agent = Drone_REINFORCE(obs_space_dims, action_space_dims)

for episode in range(total_num_episodes):
    obs = drone_env.reset()

    done = False
    while not done:
        action = drone_agent.sample_action(obs)
        obs, reward, terminated, truncated = drone_env.step(action)
        drone_agent.rewards.append(reward)
        done = terminated or truncated
    
    drone_agent.update()
    print_neural_net_parameters("Shared Net Parameters: " + str(drone_agent.net.shared_net))
    print_neural_net_parameters("Means Net Parameters: " + str(drone_agent.net.policy_mean_net))
    print_neural_net_parameters("StdDevs Net Parameters: " + str(drone_agent.net.policy_stddev_net))

drone_env.shutdown()