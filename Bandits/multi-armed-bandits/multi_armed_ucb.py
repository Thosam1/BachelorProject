import numpy as np
from multi_armed_environment import *

class MultiArmedUCB:
    def __init__(self, n_arms, n_steps, c):
        """
        Initializes the MultiArmedUCB class.

        Args:
            n_arms (int): The number of arms in the multi-armed bandit.
            n_steps (int): The total number of steps in the experiment.
            c (float): The exploration-exploitation trade-off parameter.

        Returns:
            None
        """
        # Initialize variables
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.c = c

        self.total_rewards_by_arm = np.zeros(n_arms)
        self.total_plays_by_arm = np.zeros(n_arms)

        self.upper_bounds = np.ones(n_arms) * float('inf')
        self.lower_bounds = np.ones(n_arms) * float('-inf')
        self.selected_arms = np.zeros(n_steps)
        self.rewards = np.zeros(n_steps)

    # Choose the best action
    def choose_action(self, curr_round):
        """
        Chooses the best action using the UCB (Upper Confidence Bound) algorithm.

        Args:
            curr_round (int): The current round of the experiment.

        Returns:
            int: The index of the arm to be pulled.
        """
        # Calculate upper confidence bounds for each arm
        for i in range(self.n_arms):
            if self.total_plays_by_arm[i] > 0:
                expected_reward_i = self.total_rewards_by_arm[i] / self.total_plays_by_arm[i]
                
                self.upper_bounds[i] = expected_reward_i + self.c / np.sqrt(self.total_plays_by_arm[i])
                self.lower_bounds[i] = expected_reward_i - self.c / np.sqrt(self.total_plays_by_arm[i])

        # Select arm with highest upper confidence bound and storing it
        selected_arm = np.argmax(self.upper_bounds)
        self.selected_arms[curr_round] = selected_arm
        return selected_arm

    def update(self, curr_round, action, reward):
        """
        Updates the total rewards and total plays of the chosen arm.

        Args:
            curr_round (int): The current round of the experiment.
            action (int): The index of the chosen arm.
            reward (float): The observed reward.

        Returns:
            None
        """
        self.rewards[curr_round] = reward
        self.total_rewards_by_arm[action] += reward
        self.total_plays_by_arm[action] += 1

def run_multi_armed_ucb(n_arms, n_steps, p, rewards_associated, c):
    """
    Runs the UCB (Upper Confidence Bound) algorithm on a multi-armed bandit environment.

    Args:
        n_arms (int): The number of arms in the multi-armed bandit.
        n_steps (int): The total number of steps in the experiment.
        p (numpy.ndarray): An array of probabilities of success for each arm.
        rewards_associated (numpy.ndarray): An array of rewards associated with each arm.
        c (float): The exploration-exploitation trade-off parameter.

    Returns:
        numpy.ndarray: The array of cumulative regrets over time.
    """
    # Initializing the ucb class
    multi_armed_ucb = MultiArmedUCB(n_arms, n_steps, c)

    # Initializing the environment
    environment = Environment(n_arms, n_steps, p, rewards_associated)

    for t in range(n_steps):
        # Picking up the best action
        arm_chosen = multi_armed_ucb.choose_action(t)

        # Observing the reward based on the chosen action
        expected_reward = environment.observe_reward(t, arm_chosen)

        # Compute regret
        environment.calculate_regret(t, arm_chosen)

        # Update algorithm values after receiving reward
        multi_armed_ucb.update(t, arm_chosen, expected_reward)

    # Get the array of cumulative regrets over time
    regrets = environment.get_regrets()

    return regrets

def run_multi_armed_ucb_average(n_simulations, n_arms, n_steps, p, rewards_associated, c):
    """
    Runs the UCB algorithm for a single c value and calculates average regrets.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms in the multi-armed bandit.
        n_steps (int): The total number of steps in the experiment.
        p (numpy.ndarray): An array of probabilities of success for each arm.
        rewards_associated (numpy.ndarray): An array of rewards associated with each arm.
        c (float): The c value to be tested.

    Returns:
        list: A list containing the average of cumulative regrets for the specified c value.
    """
    regrets = []

    total_regret = 0

    # Run simulations
    for simulation in range(n_simulations):
        total_regret += run_multi_armed_ucb(n_arms, n_steps, p, rewards_associated, c)

    # Calculate average regret
    avg_regret = total_regret / n_simulations

    # Append average regret to the list
    regrets.append(avg_regret)

    return regrets


def run_multi_armed_ucb_average_for_different_c_values(n_simulations, n_arms, n_steps, p, rewards_associated, c_list):
    """
    Runs the UCB algorithm for different c values and calculates average regrets.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms in the multi-armed bandit.
        n_steps (int): The total number of steps in the experiment.
        p (numpy.ndarray): An array of probabilities of success for each arm.
        rewards_associated (numpy.ndarray): An array of rewards associated with each arm.
        c_list (list): A list of c values to be tested.

    Returns:
        list: A list containing the average of cumulative regrets for each c value.
    """
    regrets = []

    # Iterate over each c value
    for c in c_list:
        regrets.extend(run_multi_armed_ucb_average(n_simulations, n_arms, n_steps, p, rewards_associated, c))

    return regrets