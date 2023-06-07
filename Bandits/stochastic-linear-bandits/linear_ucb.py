import numpy as np
from linear_bandits_environment import *

class LinUCB:
    def __init__(self, n_arms, n_features, item_features, n_rounds, lambda_param, beta_fixed=True, beta_value=1.0):
        """
        Initializes the LinUCB algorithm.

        Args:
            n_arms (int): The number of arms in the multi-armed bandit.
            n_features (int): The number of features in the item vectors.
            item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
            n_rounds (int): The total number of rounds in the experiment.
            lambda_param (float): The regularization parameter.
            beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
            beta_value (float): The fixed beta parameter value.

        Returns:
            None
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.item_features = item_features
        self.n_rounds = n_rounds
        self.lambda_param = lambda_param
        self.beta_fixed = beta_fixed
        self.beta_value = beta_value

        # Initializing variables
        self.V_t = lambda_param * np.eye(n_features)
        self.sum_A_s_X_s = np.zeros(n_features)
        self.theta_hat = np.zeros((n_features, n_rounds + 1))
        self.theta_hat[:, 0] = np.random.uniform(low=-1, high=1, size=(n_features,)).reshape((n_features,))

        self.beta_param_t = beta_value
        self.actions = np.zeros(n_rounds + 1, dtype=int)
        self.rewards = np.zeros(n_rounds + 1)

    def choose_action(self, curr_round):
        """
        Chooses the best action based on the last estimated theta_hat.

        Args:
            curr_round (int): The current round of the experiment.

        Returns:
            int: The index of the best action.
        """
        if not self.beta_fixed:
            # The beta parameter increases logarithmically with rounds
            self.beta_param_t = np.log(curr_round + 1.0) + 1.0

        max_value = -np.Inf
        max_index = -1
        for i in range(self.n_arms):
            # Compute the UCB for each arm using the given formula
            estimated_value = self.theta_hat[:, curr_round - 1].T @ self.item_features[:, i]
            penalty_value = np.sqrt(self.beta_param_t) * np.sqrt(
                self.item_features[:, i].T @ (np.linalg.inv(self.V_t) @ self.item_features[:, i]))
            value = estimated_value + penalty_value
            if value >= max_value:
                max_value = value
                max_index = i

        # Store the chosen action
        self.actions[curr_round] = max_index
        return max_index

    def update(self, curr_round, reward):
        """
        Updates the variables after receiving the reward from an action.

        Args:
            curr_round (int): The current round of the experiment.
            reward (float): The observed reward.

        Returns:
            None
        """
        self.rewards[curr_round] = reward

        # Update the feature matrix V_t by adding the outer product of the chosen action's feature vector with itself.
        self.V_t += np.outer(self.item_features[:, self.actions[curr_round]],
                             self.item_features[:, self.actions[curr_round]])

        # Compute the inverse of the updated feature matrix V_t.
        V_t_inv = np.linalg.inv(self.V_t)

        # Update the label vector sum_A_s_X_s by adding the outer product of the chosen action's feature vector with the observed reward.
        self.sum_A_s_X_s += self.item_features[:, self.actions[curr_round]] * self.rewards[curr_round]

        # Compute the new estimate of the true_theta vector using the updated feature matrix and label vector.
        # This estimate represents the center of the ellipsoid in the feature space.
        self.theta_hat[:, curr_round] = V_t_inv @ self.sum_A_s_X_s

    # Returns the matrix of each estimated theta at every round
    def get_all_theta_hat(self):
        """
        Returns the matrix of each estimated theta at every round.

        Returns:
            numpy.ndarray: The matrix of estimated theta at every round with shape (n_features, n_rounds+1).
        """
        return self.theta_hat

    # Returns the last estimated theta
    def get_last_theta_hat(self):
        """
        Returns the last estimated theta.

        Returns:
            numpy.ndarray: The last estimated theta with shape (n_features,).
        """
        return self.theta_hat[:, self.n_rounds + 1]

def run_lin_ucb(n_arms, n_features, item_features, n_rounds, true_theta, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Runs a simulation of the Linear UCB algorithm.

    Args:
        n_arms (int): The number of arms in the multi-armed bandit.
        n_features (int): The number of features in the item vectors.
        item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
        n_rounds (int): The total number of rounds in the experiment.
        true_theta (numpy.ndarray): The true theta vector with shape (n_features,).
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        numpy.ndarray: The cumulative regrets over the rounds.
        numpy.ndarray: The matrix of estimated theta at every round with shape (n_features, n_rounds+1).
    """
    # Initializing the LinUCB class
    linucb = LinUCB(n_arms, n_features, item_features, n_rounds, lambda_param, beta_fixed, beta_value)

    # Initializing the environment
    environment = Environment(n_arms, n_features, item_features, n_rounds, true_theta, noise)

    for t in range(1, n_rounds + 1):
        # Picking the best action based on theta_hat
        arm_chosen = linucb.choose_action(t)

        # Reward received based on the action taken
        expected_reward, reward_with_noise = environment.observe_reward(t, arm_chosen)

        # Compute regret
        environment.calculate_regret(t, expected_reward)

        # Update algorithm values after receiving reward
        linucb.update(t, reward_with_noise)

    regrets = environment.get_regrets()
    all_theta_hat = linucb.get_all_theta_hat()

    return regrets, all_theta_hat

def run_lin_ucb_average(n_simulations, n_arms, n_features, n_rounds, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Runs the Linear UCB algorithm multiple times and calculates the average regrets, theta_hat, and difference per round.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms in the multi-armed bandit.
        n_features (int): The number of features in the item vectors.
        n_rounds (int): The total number of rounds in the experiment.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        numpy.ndarray: The average cumulative regrets over the rounds.
        numpy.ndarray: The average matrix of estimated theta at every round with shape (n_features, n_rounds+1).
        numpy.ndarray: The average difference between true theta and estimated theta per round.
    """
    total_regrets = np.zeros(n_rounds + 1)
    total_all_theta_hat = np.zeros((n_features, n_rounds + 1))
    total_diff_per_round = np.zeros(n_rounds + 1)

    for i in range(n_simulations):
        # Generating new values of true theta and item features
        item_features = np.random.uniform(low=-1, high=1, size=(n_features, n_arms))
        true_theta = np.random.uniform(low=-1, high=1, size=(n_features, 1))

        curr_regret_array, curr_all_theta_hat = run_lin_ucb(n_arms, n_features, item_features, n_rounds, true_theta, noise, lambda_param, beta_fixed, beta_value)
        
        total_regrets += curr_regret_array
        total_all_theta_hat += curr_all_theta_hat
        total_diff_per_round += diff_theta_hat_true_theta(true_theta, curr_all_theta_hat)

    avg_regret = total_regrets / n_simulations
    avg_curr_all_theta_hat = total_all_theta_hat / n_simulations
    avg_total_diff_per_round = total_diff_per_round / n_simulations

    return avg_regret, avg_curr_all_theta_hat, avg_total_diff_per_round

def diff_theta_hat_true_theta(true_theta, theta_hat_array):
    """
    Calculates the difference between true theta and estimated theta per round.

    Args:
        true_theta (numpy.ndarray): The true theta vector with shape (n_features, 1).
        theta_hat_array (numpy.ndarray): The matrix of estimated theta at every round with shape (n_features, n_rounds+1).

    Returns:
        numpy.ndarray: The array of differences between true theta and estimated theta per round.
    """
    n_features = theta_hat_array.shape[0]
    n_rounds = theta_hat_array.shape[1]
    diff_per_round = np.zeros(n_rounds)

    for i in range(n_rounds):
        # Calculate the absolute difference between true theta and estimated theta for each feature
        diff = np.abs(np.subtract(true_theta, theta_hat_array[:, i].reshape(n_features, 1)))
        
        # Calculate the total difference for the round by summing the absolute differences for all features
        total_diff = np.sum(diff)
        
        # Store the total difference for the round in the diff_per_round array
        diff_per_round[i] = total_diff

    return diff_per_round
