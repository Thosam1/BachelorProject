import numpy as np

class Environment:
    def __init__(self, n_arms, n_steps, p, rewards_associated):
        """
        Initializes the Environment class.

        Args:
            n_arms (int): The number of arms in the multi-armed bandit.
            n_steps (int): The total number of steps in the experiment.
            p (numpy.ndarray): An array of probabilities of success for each arm.
            rewards_associated (numpy.ndarray): An array of rewards associated with each arm.

        Returns:
            None
        """
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.p = p
        self.rewards_associated = rewards_associated
        self.rewards = np.zeros(n_steps)
        self.regrets = np.zeros(n_steps)

        self.optimal_expected_reward = -np.Inf
        
        for i in range(n_arms): 
            expected_reward = p[i] * rewards_associated[i]
            if(expected_reward > self.optimal_expected_reward):
                self.optimal_expected_reward = expected_reward

    # Observe the reward of the chosen action
    def observe_reward(self, curr_round, action):
        """
        Observes the reward of the chosen action.

        Args:
            curr_round (int): The current round of the experiment.
            action (int): The index of the chosen arm.

        Returns:
            float: The expected reward.
        """
        expected_reward = 0
        if(np.random.rand() < self.p[action]):
            expected_reward = self.rewards_associated[action]

        # we could add noise to it
        self.rewards[curr_round] = expected_reward
        return expected_reward

    def calculate_regret(self, curr_round, action):
        """
        Calculates the regret at the current round and updates the cumulative regret internally.

        Args:
            curr_round (int): The current round of the experiment.
            action (int): The index of the chosen arm.

        Returns:
            None
        """
        regret = self.optimal_expected_reward - self.p[action] * self.rewards_associated[action] # loss that can incurred by not choosing the arm with the highest expected reward
        self.regrets[curr_round] = self.regrets[curr_round - 1] + regret

    def get_regrets(self):
        """
        Returns the array of cumulative regrets over time.

        Returns:
            numpy.ndarray: The array of regrets over time.
        """
        return self.regrets