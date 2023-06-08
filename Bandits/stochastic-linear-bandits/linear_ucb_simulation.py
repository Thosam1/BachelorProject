"""
    This file contains the related functions for running simulations with the Linear UCB algorithm and
    calculating distances between true theta and estimated theta (evaluating the performance).
"""

import numpy as np
from environment import *
from linucb_algorithm import *

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
    Runs the Linear UCB algorithm multiple times and calculates the average regrets, average true theta, and average estimated theta per round.

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
        numpy.ndarray: The average true theta vector with shape (n_features, 1).
        numpy.ndarray: The average matrix of estimated theta at every round with shape (n_features, n_rounds+1).
    """
    total_regrets = np.zeros(n_rounds + 1)
    total_true_theta = np.zeros((n_features, 1))
    total_all_theta_hat = np.zeros((n_features, n_rounds + 1))

    for i in range(n_simulations):
        # Generating new values of true theta and item features
        item_features = np.random.uniform(low=-1, high=1, size=(n_features, n_arms))
        true_theta = np.random.uniform(low=-1, high=1, size=(n_features, 1))

        curr_regret_array, curr_all_theta_hat = run_lin_ucb(n_arms, n_features, item_features, n_rounds, true_theta, noise, lambda_param, beta_fixed, beta_value)
        
        total_regrets += curr_regret_array
        total_true_theta += true_theta
        total_all_theta_hat += curr_all_theta_hat

    avg_regret = total_regrets / n_simulations
    avg_true_theta = total_true_theta / n_simulations
    avg_curr_all_theta_hat = total_all_theta_hat / n_simulations

    return avg_regret, avg_true_theta, avg_curr_all_theta_hat

def run_lin_ucb_average_multiple_arms(n_simulations, n_arms_list, n_features, n_rounds, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Runs the Linear UCB algorithm multiple times for different numbers of arms and calculates the average regrets,
    average true theta, and average estimated theta per round.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms_list (list): A list of numbers of arms in the multi-armed bandit for each simulation.
        n_features (int): The number of features in the item vectors.
        n_rounds (int): The total number of rounds in the experiment.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        list: A list of tuples containing the average cumulative regrets over the rounds, the average true theta vector,
              and the average matrix of estimated theta at every round for each simulation.
    """
    results = []
    for n_arms in n_arms_list:
        result = run_lin_ucb_average(n_simulations, n_arms, n_features, n_rounds, noise, lambda_param, beta_fixed, beta_value)
        results.append(result)
    return results

def run_lin_ucb_average_multiple_features(n_simulations, n_arms, n_features_list, n_rounds, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Runs the Linear UCB algorithm multiple times for different numbers of features and calculates the average regrets,
    average true theta, and average estimated theta per round.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms in the multi-armed bandit.
        n_features_list (list): A list of numbers of features in the item vectors for each simulation.
        n_rounds (int): The total number of rounds in the experiment.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        list: A list of tuples containing the average cumulative regrets over the rounds, the average true theta vector,
              and the average matrix of estimated theta at every round for each simulation.
    """
    results = []
    for n_features in n_features_list:
        result = run_lin_ucb_average(n_simulations, n_arms, n_features, n_rounds, noise, lambda_param, beta_fixed, beta_value)
        results.append(result)
    return results

def theta_hat_true_theta_distance(true_theta, theta_hat_array, distance_function):
    """
    Calculates the distance between true theta and estimated theta per round.

    Args:
        true_theta (numpy.ndarray): The true theta vector with shape (n_features, 1).
        theta_hat_array (numpy.ndarray): The matrix of estimated theta at every round with shape (n_features, n_rounds+1).
        distance_function (function): The distance function to calculate the distance between two vectors.

    Returns:
        numpy.ndarray: The array of distances between true theta and estimated theta per round.
    """
    n_features = theta_hat_array.shape[0]
    n_rounds = theta_hat_array.shape[1]
    distance_per_round = np.zeros(n_rounds)

    for i in range(n_rounds):
        distance_per_round[i] = distance_function(true_theta, theta_hat_array[:, i].reshape(n_features, 1))

    return distance_per_round