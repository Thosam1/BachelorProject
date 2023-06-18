"""
    This file contains the related functions for running simulations with the Linear UCB algorithm (both training and testing) and
    calculating distances between true theta and estimated theta (evaluating the performance).
"""

from linucb_training import *
from linucb_testing import *

def simulation_on_one_user(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Performs a simulation for a single user using the linear UCB bandit algorithm.

    Args:
        features_by_music_id_dict (dict): A dictionary mapping music IDs to their corresponding feature vectors.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.
        selected_reviewer_id (str): The ID of the selected reviewer for simulation.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The number of rounds for each simulation.
        max_reward (float): The maximum possible reward value.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the training results and the evaluation results for the selected user. The training results include the average regret, all theta hat values, the last theta hat, and the last V_t matrix. The evaluation results include the average difference between true and estimated rewards, a list of true and estimated reward values for each arm, and the total number of test samples.
    """
    selected_reviewer_training_results = train_linear_ucb_for_one_user(features_by_music_id_dict, ratings_by_reviewer_training, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_value)
    selected_reviewer_absolute_diff = test_ucb_linear_bandit_for_one_user(selected_reviewer_training_results, features_by_music_id_dict, ratings_by_reviewer_test, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_value)

    return (selected_reviewer_training_results, selected_reviewer_absolute_diff)

def simulation_on_all_users(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed=True, beta_value=1.0):
    """
    Performs simulations for all users using the linear UCB bandit algorithm.

    Args:
        features_by_music_id_dict (dict): A dictionary mapping music IDs to their corresponding feature vectors.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The number of rounds for each simulation.
        max_reward (float): The maximum possible reward value.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_value (float): The fixed beta parameter value.

    Returns:
        tuple: A tuple containing the average regrets and the average difference per dimension. The average regrets represent the average regret values across all users, and the average difference represents the average difference between true and estimated rewards per dimension across all users.
    """
    total_regrets = 0

    # To have weighted avg because all users do not have the same nb of test ratings
    total_diff = 0. # of all users
    total_nb_ratings = 0. # of all users

    for selected_reviewer_id in ratings_by_reviewer_training:
        # Running the algorithm for a given user
        selected_reviewer_training_results, selected_reviewer_absolute_diff = simulation_on_one_user(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, selected_reviewer_id, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_value)

        # Results for regrets for dimenstions as list
        total_regrets += selected_reviewer_training_results[0] # todo weighted
 
        avg_diff_selected_reviewer = selected_reviewer_absolute_diff[0]
        total_nb_test_reviews_of_selected_reviewer = selected_reviewer_absolute_diff[2]

        total_diff += avg_diff_selected_reviewer * total_nb_test_reviews_of_selected_reviewer
        total_nb_ratings += total_nb_test_reviews_of_selected_reviewer

    # Getting the regret averages (to analyse training)
    avg_regrets = total_regrets / len(list(ratings_by_reviewer_training.keys())) # todo weighted

    # Getting the average diff (to analyse testing)
    avg_diff = total_diff / total_nb_ratings

    return (avg_regrets, avg_diff)

def simulation_on_all_users_all_dimensions(features_dict_by_dim, ratings_by_reviewer_training, ratings_by_reviewer_test, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_values):
    """
    Performs simulations for all users and all dimensions using the linear UCB bandit algorithm for different values of the beta parameter.

    Args:
        features_dict_by_dim (dict): A dictionary mapping dimension keys to features dictionaries by music ID.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The number of rounds for each simulation.
        max_reward (float): The maximum possible reward value.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_values (list): A list of beta values to be used for simulations.

    Returns:
        tuple: A tuple containing the lists of dictionaries with average regrets and average difference per dimension for all the beta values. The average regrets by dimension list contains dictionaries mapping dimension keys to average regret values across all users for each beta value, and the average difference by dimension list contains dictionaries mapping dimension keys to average difference between true and estimated rewards per dimension across all users for each beta value.
    """
    # To store for each value of beta
    avg_regrets_by_dim_list = []
    avg_diff_by_dim_list = []

    for beta in beta_values:
        avg_regrets_by_dim = {}
        avg_diff_by_dim = {}

        for key in features_dict_by_dim:
            # for music
            features_by_music_id_dict = features_dict_by_dim[key]
            avg_regrets, avg_diff = simulation_on_all_users(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta)

            # Storing the results
            avg_regrets_by_dim[key] = avg_regrets
            avg_diff_by_dim[key] = avg_diff

        avg_regrets_by_dim_list.append(avg_regrets_by_dim)
        avg_diff_by_dim_list.append(avg_diff_by_dim)

    return avg_regrets_by_dim_list, avg_diff_by_dim_list

def simulation_on_all_users_for_different_betas(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta_values):
    """
    Performs simulations for all users using the linear UCB bandit algorithm for different values of the beta parameter.

    Args:
        features_by_music_id_dict (dict): A dictionary mapping music IDs to their corresponding feature vectors.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.
        n_simulations (int): The number of simulations to run.
        n_rounds (int): The number of rounds for each simulation.
        max_reward (float): The maximum possible reward value.
        noise (float): The standard deviation of the noise added to the rewards.
        lambda_param (float): The regularization parameter.
        beta_fixed (bool): A flag indicating whether the beta parameter is fixed or varies with rounds.
        beta_values (list): A list of beta values to be used for simulations.

    Returns:
        tuple: A tuple containing the lists of average regrets and average difference per dimension for all the beta values. The average regrets list contains the average regret values across all users for each beta value, and the average difference list contains the average difference between true and estimated rewards per dimension across all users for each beta value.
    """
    avg_regrets_list = []
    avg_diff_list = []

    for beta in beta_values:
        avg_regrets, avg_diff = simulation_on_all_users(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, n_simulations, n_rounds, max_reward, noise, lambda_param, beta_fixed, beta)
        
        avg_regrets_list.append(avg_regrets)
        avg_diff_list.append(avg_diff)

    return avg_regrets_list, avg_diff_list
