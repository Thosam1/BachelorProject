"""
    This file contains the related functions to run linear regression in order to compare its effectiveness with linear bandits.
"""

import sys
sys.path.append('../utils/')
from data_processing import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def linear_regression_on_one_user(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, selected_reviewer_id):
    """
    Performs a linear regression for a single user.

    Args:
        features_by_music_id_dict (dict): A dictionary mapping music IDs to their corresponding feature vectors.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.
        selected_reviewer_id (str): The ID of the selected reviewer for simulation.

    Returns:
        tuple: A tuple containing the number of test samples and the average absolute difference between true and estimated rewards.
    """
    n_features = len(features_by_music_id_dict[next(iter(features_by_music_id_dict))])

    # Training phase
    item_features, item_rewards, _ = get_item_features_and_rewards_for_user(n_features, ratings_by_reviewer_training[selected_reviewer_id], features_by_music_id_dict)
    X = np.transpose(item_features)
    y = item_rewards.reshape((item_rewards.shape[0], 1))

    # Create an instance of the Linear Regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X, y)

    # Testing phase
    item_features, item_rewards, _ = get_item_features_and_rewards_for_user(n_features, ratings_by_reviewer_test[selected_reviewer_id], features_by_music_id_dict)
    X_test = np.transpose(item_features)
    y_test = item_rewards.reshape((item_rewards.shape[0], 1))

    # Predict the ratings for the test data
    predicted_ratings = model.predict(X_test)

    # Calculate the number of test samples
    nb_test_samples = item_rewards.shape[0]

    # Calculate the average absolute difference between true and estimated rewards
    absolute_difference = np.sum(np.abs(np.subtract(predicted_ratings, y_test)))

    return nb_test_samples, absolute_difference

def linear_regression_on_all_users(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test):
    """
    Performs linear regression on all users and calculates the average absolute difference between true and estimated rewards.

    Args:
        features_by_music_id_dict (dict): A dictionary mapping music IDs to their corresponding feature vectors.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.

    Returns:
        float: The average absolute difference between true and estimated rewards across all users.
    """
    # Variables for weighted average
    total_overall_absolute_diff = 0.0
    total_overall_nb_samples = 0

    for selected_reviewer_id in ratings_by_reviewer_training:
        # Perform linear regression for each user
        nb_test_samples, total_absolute_diff = linear_regression_on_one_user(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test, selected_reviewer_id)

        # Accumulate the total absolute difference and number of test samples
        total_overall_absolute_diff += total_absolute_diff
        total_overall_nb_samples += nb_test_samples

    # Calculate the average absolute difference
    average_absolute_diff = total_overall_absolute_diff / total_overall_nb_samples

    return average_absolute_diff

def linear_regression_on_all_users_all_dimensions(features_dict_by_dim, ratings_by_reviewer_training, ratings_by_reviewer_test):
    """
    Performs linear regression on all users for each dimension of the feature vectors and calculates the average absolute difference between true and estimated rewards.

    Args:
        features_dict_by_dim (dict): A dictionary mapping dimensions to a dictionary of features by music ID for that dimension.
        ratings_by_reviewer_training (dict): A dictionary mapping reviewer IDs to their ratings for training music IDs.
        ratings_by_reviewer_test (dict): A dictionary mapping reviewer IDs to their ratings for test music IDs.

    Returns:
        list: A list of dictionaries containing the average absolute difference by dimension.
    """
    avg_diff_by_dim_list = []
    # Store the average absolute difference by dimension
    avg_diff_by_dim = {}

    for key in features_dict_by_dim:
        # Get features for the specific dimension
        features_by_music_id_dict = features_dict_by_dim[key]
        
        # Perform linear regression for all users in the specific dimension
        avg_diff = linear_regression_on_all_users(features_by_music_id_dict, ratings_by_reviewer_training, ratings_by_reviewer_test)
        avg_diff_by_dim[key] = avg_diff
        
    avg_diff_by_dim_list.append(avg_diff_by_dim)

    return avg_diff_by_dim_list
