"""
    This file contains the related functions related to testing the Linear UCB algorithm.
"""

import numpy as np
import sys
sys.path.append('../utils/')
from data_processing import *

def lin_ucb_choose_action_test(curr_round, theta_hat, V_t, n_arms, item_features, beta_fixed=True, beta_value=1.0):
    """
    Simulates the process of choosing an action based on the given parameters for testing purposes.

    Args:
        curr_round (int): The current round of the experiment.
        theta_hat (numpy.ndarray): The estimated theta vector with shape (n_features,).
        V_t (numpy.ndarray): The V_t matrix with shape (n_features, n_features).
        n_arms (int): The number of arms in the multi-armed bandit.
        item_features (numpy.ndarray): The matrix of item features with shape (n_features, n_arms).
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the index of the best action, the corresponding value, and a list of indices and values for all actions.
    """
    beta_param_t = beta_value
    if(beta_fixed==False):
        # must increase logarithmically
        beta_param_t = np.log(curr_round + 1.0) + 1.0

    indices_values = []
    max_value = -np.Inf
    max_index = -1
    for i in range(n_arms):
        # Compute the UCB for each arm using the given formula
        estimated_value = theta_hat.T @ item_features[:, i]
        penalty_value = np.sqrt(beta_param_t) * np.sqrt(
            item_features[:, i].T @ (np.linalg.inv(V_t) @ item_features[:, i]))
        value = estimated_value # + penalty_value
        indices_values.append((i, value))
        if value >= max_value:
            max_value = value
            max_index = i

    return max_index, max_value, indices_values

def test_ucb_linear_bandit_for_one_user(training_results, features_by_music_id_dict, ratings_by_reviewer, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Tests the performance of the linear UCB bandit algorithm for a single user.

    Args:
        training_results (tuple): A tuple containing the training results, including average regret, all theta hat values, last theta hat, and last V_t matrix.
        features_by_music_id_dict (dict): A dictionary mapping music IDs to their feature vectors.
        ratings_by_reviewer (dict): A dictionary mapping reviewer IDs to their ratings for different music IDs.
        selected_reviewer_id (str): The ID of the selected reviewer for testing.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The number of rounds for the test.
        max_reward (float): The maximum possible reward value.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the evaluation results, including the average difference between true and estimated rewards, a list of true and estimated reward values for each arm, and the total number of test samples.
    """
    n_arms = len(ratings_by_reviewer[selected_reviewer_id].keys())
    n_features = len(features_by_music_id_dict[list(features_by_music_id_dict.keys())[0]])
    item_features, item_rewards, items_order = get_item_features_and_rewards_for_user(n_features, ratings_by_reviewer[selected_reviewer_id], features_by_music_id_dict)

    # Getting the V_t and the last theta hat
    last_theta_hat = training_results[2]
    last_v_t = training_results[3]
    max_index, max_value, indices_values = lin_ucb_choose_action_test(n_rounds+1, last_theta_hat, last_v_t, n_arms, item_features, beta_fixed, beta_value)

    # Returning necessary values for test results plots
    list_true_estimated = []
    absolute_diff = 0
    for tuple in indices_values:
        index = tuple[0]
        value = tuple[1]
        true_reward = ratings_by_reviewer[selected_reviewer_id][items_order[index]]
        absolute_diff += abs(float(true_reward) - float(value))
        list_true_estimated.append((float(true_reward), value))

    # First element is the average of all differences of all the test samples for that particular user (because nb of test samples varies between users)
    total_test_samples = len(indices_values)
    return (absolute_diff / float(total_test_samples), list_true_estimated, total_test_samples)

def test_ucb_linear_bandit_for_one_user_all_dimensions(training_results_by_dim, features_dict_by_dim, ratings_by_reviewer, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Tests the performance of the linear UCB bandit algorithm for a single user across multiple dimensions.

    Args:
        training_results_by_dim (dict): A dictionary mapping dimensions to their corresponding training results, including average regret, all theta hat values, last theta hat, and last V_t matrix.
        features_dict_by_dim (dict): A dictionary mapping dimensions to their corresponding features dictionary by music ID.
        ratings_by_reviewer (dict): A dictionary mapping reviewer IDs to their ratings for different music IDs.
        selected_reviewer_id (str): The ID of the selected reviewer for testing.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The number of rounds for the test.
        max_reward (float): The maximum possible reward value.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        dict: A dictionary mapping dimensions to their corresponding evaluation results, including the average difference between true and estimated rewards, a list of true and estimated reward values for each arm, and the total number of test samples.
    """
    absolute_diff_by_dim = {}

    # Running the algorithm for all dimensions
    for key in features_dict_by_dim:
        # Getting the dictionary with music id as key and np array of features as value
        features_by_music_id = features_dict_by_dim[key]
        absolute_diff_by_dim[key] = test_ucb_linear_bandit_for_one_user(training_results_by_dim[key], features_by_music_id, ratings_by_reviewer, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_value)

    return absolute_diff_by_dim