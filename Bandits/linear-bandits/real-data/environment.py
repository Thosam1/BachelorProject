"""
    This file contains an adaptation of the Environment class made for training the linear bandits on the real data from the Amazon music dataset, which represents the environment for a linear
    bandit algorithm. It provides methods for initializing the environment, observing rewards, calculating regrets,
    and accessing cumulative regrets.
"""

import numpy as np

class EnvironmentLinUCB:
    def __init__(self, n_arms, n_features, item_features, n_rounds, noise, max_reward, item_rewards):
        """
        Initializes the environment for the LinUCB algorithm.

        Args:
            n_arms (int): The number of arms in the multi-armed bandit.
            n_features (int): The number of features in the item vectors.
            item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
            n_rounds (int): The total number of rounds in the experiment.
            noise (float): The standard deviation of the noise added to the rewards.
            max_reward (float): The maximum possible reward in the environment.
            item_rewards (numpy.ndarray): The array of true rewards associated with each arm.

        Returns:
            None
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.item_features = item_features
        self.n_rounds = n_rounds
        self.noise = noise
        self.item_rewards = item_rewards

        # Initializing variables
        self.optimal_reward = max_reward
        self.rewards = np.zeros(n_rounds + 1)  # rewards with noise
        self.regrets = np.zeros(n_rounds + 1)

    def observe_reward(self, curr_round, action):
        """
        Observes the reward of the chosen action and adds noise.

        Args:
            curr_round (int): The current round of the experiment.
            action (int): The index of the chosen action.

        Returns:
            float: The expected reward without noise.
            float: The observed reward with added noise.
        """
        expected_reward = self.item_rewards[action]
        self.rewards[curr_round] = expected_reward + np.random.normal(scale=self.noise)
        return expected_reward, self.rewards[curr_round]

    def calculate_regret(self, curr_round, expected_reward):
        """
        Computes the regret for the current round.

        Args:
            curr_round (int): The current round of the experiment.
            expected_reward (float): The expected reward without noise.

        Returns:
            None
        """
        regret = self.optimal_reward - expected_reward
        self.regrets[curr_round] = self.regrets[curr_round - 1] + regret

    def get_regrets(self):
        """
        Returns the cumulative regrets.

        Returns:
            numpy.ndarray: The array of cumulative regrets.
        """
        return self.regrets
