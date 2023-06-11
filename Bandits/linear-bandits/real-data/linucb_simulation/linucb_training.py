"""
    This file contains the related functions related to training the Linear UCB algorithm.
"""

from environment import *
import sys
sys.path.append('../utils/')
from data_processing import *
sys.path.append('../')
from linucb_algorithm import *

def run_lin_ucb(n_arms, n_features, item_features, n_rounds, max_reward, item_rewards, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Runs a simulation of the Linear UCB algorithm.

    Args:
        n_arms (int): The number of arms (actions) in the multi-armed bandit.
        n_features (int): The number of features in the item vectors.
        item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
        n_rounds (int): The total number of rounds in the simulation.
        max_reward (float): The maximum possible reward.
        item_rewards (numpy.ndarray): The array of rewards for each item.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The lambda parameter for the LinUCB algorithm.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the regrets, all theta_hat values, the last theta_hat, and the last V_t matrix.
    """
    linucb = LinUCB(n_arms, n_features, item_features, n_rounds, lambda_param, beta_fixed, beta_value)
    environment = EnvironmentLinUCB(n_arms, n_features, item_features, n_rounds, noise, max_reward, item_rewards)

    for t in range(1, n_rounds + 1):
        arm_chosen = linucb.choose_action(t)
        expected_reward, reward_with_noise = environment.observe_reward(t, arm_chosen)
        environment.calculate_regret(t, expected_reward)
        linucb.update(t, reward_with_noise)

    regrets = environment.get_regrets()
    all_theta_hat = linucb.get_all_theta_hat()
    last_theta_hat = linucb.get_last_theta_hat()
    last_v_t = linucb.get_last_v_t()

    return regrets, all_theta_hat, last_theta_hat, last_v_t

def run_lin_ucb_average(n_simulations, n_arms, n_features, n_rounds, max_reward, item_rewards, item_features, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Runs multiple simulations of the Linear UCB algorithm and returns the averaged results.

    Args:
        n_simulations (int): The number of simulations to run.
        n_arms (int): The number of arms (actions) in the multi-armed bandit.
        n_features (int): The number of features in the item vectors.
        n_rounds (int): The total number of rounds in each simulation.
        max_reward (float): The maximum possible reward.
        item_rewards (numpy.ndarray): The array of rewards for each item.
        item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The lambda parameter for the LinUCB algorithm.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the averaged regrets, all averaged theta_hat values, the averaged last theta_hat, and the averaged last V_t matrix.
    """
    total_regrets = np.zeros(n_rounds + 1)
    total_all_theta_hat = np.zeros((n_features, n_rounds + 1))
    total_last_theta_hat = np.zeros(n_features)
    total_last_v_t = np.eye(n_features)

    for i in range(n_simulations):
        curr_regret_array, curr_all_theta_hat, curr_last_theta_hat, curr_last_v_t = run_lin_ucb(n_arms, n_features, item_features, n_rounds, max_reward, item_rewards, noise, lambda_param, beta_fixed, beta_value)
        total_regrets += curr_regret_array
        total_all_theta_hat += curr_all_theta_hat
        total_last_theta_hat += curr_last_theta_hat
        total_last_v_t += curr_last_v_t

    avg_regret = total_regrets / n_simulations
    avg_curr_all_theta_hat = total_all_theta_hat / n_simulations
    avg_last_theta_hat = total_last_theta_hat / n_simulations
    avg_last_v_t = total_last_v_t / n_simulations

    return avg_regret, avg_curr_all_theta_hat, avg_last_theta_hat, avg_last_v_t

def train_linear_ucb_for_one_user(features_by_music_id_dict, ratings_by_reviewer, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Trains the Linear UCB algorithm for a single user.

    Args:
        features_by_music_id_dict (dict): A dictionary containing the feature vectors for each music. The music ID should be the key, and the feature vector should be a numpy array.
        ratings_by_reviewer (dict): A dictionary containing the ratings for each reviewer. The reviewer ID should be the key, and the ratings should be a dictionary with the music ID as the key and the rating as the value.
        selected_reviewer_id (str): The ID of the selected reviewer to train the algorithm for.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The total number of rounds in each simulation.
        max_reward (float): The maximum possible reward.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the average regret, average estimated theta at each round, average last estimated theta, and average last V_t matrix.
    """
    n_arms = len(ratings_by_reviewer[selected_reviewer_id].keys())
    n_features = len(features_by_music_id_dict[list(features_by_music_id_dict.keys())[0]])
    item_features, item_rewards, _ = get_item_features_and_rewards_for_user(n_features, ratings_by_reviewer[selected_reviewer_id], features_by_music_id_dict)

    avg_regret, avg_all_theta_hat, avg_last_theta, avg_last_v_t = run_lin_ucb_average(n_simulations, n_arms, n_features, n_rounds, max_reward, item_rewards, item_features, noise, lambda_param, beta_fixed, beta_value)
    return (avg_regret, avg_all_theta_hat, avg_last_theta, avg_last_v_t)

def train_linear_ucb_for_one_user_all_dimensions(features_dict_by_dim, ratings_by_reviewer, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Trains the Linear UCB algorithm for a single user in all dimensions.

    Args:
        features_dict_by_dim (dict): A dictionary containing the feature vectors for each dimension. The dimension ID should be the key, and the value should be a dictionary containing the feature vectors for each music. The music ID should be the key, and the feature vector should be a numpy array.
        ratings_by_reviewer (dict): A dictionary containing the ratings for each reviewer. The reviewer ID should be the key, and the ratings should be a dictionary with the music ID as the key and the rating as the value.
        selected_reviewer_id (str): The ID of the selected reviewer to train the algorithm for.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The total number of rounds in each simulation.
        max_reward (float): The maximum possible reward.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        dict: A dictionary containing the results for each dimension. The dimension ID is the key, and the value is a tuple containing the average regret, average estimated theta at each round, average last estimated theta, and average last V_t matrix.
    """
    results_by_dimension = {}

    # Running the algorithm for all dimensions
    for key in features_dict_by_dim:
        # Getting the dictionary with music id as key and np array of features as value
        features_by_music_id = features_dict_by_dim[key]
        results_by_dimension[key] = train_linear_ucb_for_one_user(features_by_music_id, ratings_by_reviewer, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_value)
    return results_by_dimension