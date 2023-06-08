"""
    This file contains the implementation of the Environment class, which represents the environment for a linear
    bandit algorithm. It provides methods for initializing the environment, observing rewards, calculating regrets,
    and accessing cumulative regrets.
"""

import numpy as np

class Environment:
    def __init__(self, n_arms, n_features, item_features, n_rounds, true_theta, noise):
        """
        Initializes the environment for a lineaer bandit algorithm.

        Args:
            n_arms (int): The number of arms in the multi-armed bandit.
            n_features (int): The number of features in the item vectors.
            item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
            n_rounds (int): The total number of rounds in the experiment.
            true_theta (numpy.ndarray): The true theta vector with shape (n_features,).
            noise (float): The standard deviation of the noise added to the rewards.

        Returns:
            None
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.item_features = item_features
        self.n_rounds = n_rounds
        self.true_theta = true_theta
        self.noise = noise

        # Initializing variables
        self.optimal_reward = np.max(item_features.T @ true_theta)
        self.rewards = np.zeros(n_rounds + 1)  # rewards with noise
        self.regrets = np.zeros(n_rounds + 1)

    # Observe the reward of the chosen action and add noise
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
        expected_reward = self.true_theta.T @ self.item_features[:, action]
        self.rewards[curr_round] = expected_reward + np.random.normal(scale=self.noise)
        return expected_reward, self.rewards[curr_round]

    # Compute the regret for this round
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

    # Returns the cumulative regret
    def get_regrets(self):
        """
        Returns the cumulative regrets.

        Returns:
            numpy.ndarray: The array of cumulative regrets.
        """
        return self.regrets
