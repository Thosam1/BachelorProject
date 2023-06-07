import numpy as np
from multi_armed_environment import *

class MultiArmedEpsilonGreedy:
    def __init__(self, n_arms, n_steps, epsilon):
        """
        Initializes the MultiArmedEpsilonGreedy class.

        Args:
            n_arms (int): The number of arms in the multi-armed bandit.
            n_steps (int): The total number of steps in the experiment.
            epsilon (float): The exploration-exploitation trade-off parameter (between 0 and 1).

        Returns:
            None
        """
        # Initialize variables
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.epsilon = epsilon

        self.Q = np.zeros(n_arms)  # Action-value estimates
        self.N = np.zeros(n_arms)  # Number of times each arm has been played

        self.actions = np.zeros(self.n_steps)  # Arm picked

    # Choose an action
    def choose_action(self, curr_round):
        """
        Chooses an action using epsilon-greedy exploration strategy.

        Args:
            curr_round (int): The current round of the experiment.

        Returns:
            int: The index of the arm to be pulled.
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_arms)
        else:
            action = np.argmax(self.Q)
        self.actions[curr_round] = action
        return action

    def update(self, action, reward):
        """
        Updates the action-value estimates based on the observed reward.

        Args:
            action (int): The index of the chosen arm.
            reward (float): The observed reward.

        Returns:
            None
        """
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) * (1 / self.N[action])

def run_epsilon_greedy(n_arms, n_steps, p, rewards_associated, epsilon):
    """
    Runs the epsilon-greedy algorithm on a multi-armed bandit environment.

    Args:
        n_arms (int): The number of arms in the multi-armed bandit.
        n_steps (int): The total number of steps in the experiment.
        p (numpy.ndarray): An array of probabilities of success for each arm.
        rewards_associated (numpy.ndarray): An array of rewards associated with each arm.
        epsilon (float): The exploration-exploitation trade-off parameter.

    Returns:
        numpy.ndarray: The array containing the average of cumulative regrets over time.
    """
    # Initializing the epsilon greedy class
    multi_armed_epsilon_greedy = MultiArmedEpsilonGreedy(n_arms, n_steps, epsilon)

    # Initializing the environment
    environment = Environment(n_arms, n_steps, p, rewards_associated)

    for t in range(n_steps):
        # Picking an action
        arm_chosen = multi_armed_epsilon_greedy.choose_action(t)

        # Reward received based on the action taken
        expected_reward = environment.observe_reward(t, arm_chosen)

        # Compute regret
        environment.calculate_regret(t, arm_chosen)

        # Update algorithm values after receiving reward
        multi_armed_epsilon_greedy.update(arm_chosen, expected_reward)

    # Get the array of regrets over time
    regrets = environment.get_regrets()

    return regrets

def run_epsilon_greedy_for_different_epsilons(n_simulations, n_arms, n_steps, p, rewards_associated, epsilons):
    """
    Runs the epsilon-greedy algorithm for different epsilon values and calculates average regrets.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms in the multi-armed bandit.
        n_steps (int): The total number of steps in the experiment.
        p (numpy.ndarray): An array of probabilities of success for each arm.
        rewards_associated (numpy.ndarray): An array of rewards associated with each arm.
        epsilons (list): A list of epsilon values to be tested.

    Returns:
        list: A list containing the average of cumulative regrets for each epsilon value.
    """
    regrets = []

    # Iterate over each epsilon value
    for epsilon in epsilons:
        total_regret = 0

        # Run simulations
        for simulation in range(n_simulations):
            total_regret += run_epsilon_greedy(n_arms, n_steps, p, rewards_associated, epsilon)

        # Calculate average regret
        avg_regret = total_regret / n_simulations

        # Append average regret to the list
        regrets.append(avg_regret)

    return regrets

def run_epsilon_greedy_average(n_simulations, n_arms, n_steps, rewards_associated, epsilon):
    """
    Runs the epsilon-greedy algorithm multiple times with randomly generated probabilities and rewards, and calculates
    the average regrets over the simulations.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms in the multi-armed bandit.
        n_steps (int): The total number of steps in the experiment.
        rewards_associated (numpy.ndarray): An array of rewards associated with each arm.
        epsilon (float): The exploration-exploitation trade-off parameter.

    Returns:
        numpy.ndarray: The array containing the average of cumulative regrets over the simulations.
    """
    total_regrets = np.zeros(n_steps)

    # Run multiple simulations
    for i in range(n_simulations):
        # Generating new values for probabilities and rewards associated
        p = np.random.rand(n_arms)
        rewards_associated = np.ones(n_arms)

        # Run epsilon-greedy algorithm for that simulation
        curr_regrets = run_epsilon_greedy(n_arms, n_steps, p, rewards_associated, epsilon)

        # Adding the current cumulative regrets array to the total
        total_regrets += curr_regrets

    # Calculate average of the cumulative regrets array over the simulations
    avg_regret = total_regrets / n_simulations

    return avg_regret
