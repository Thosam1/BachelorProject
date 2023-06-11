"""
    This file contains the related functions to handle and process the data for the Linear UCB algorithm.
"""

import numpy as np

def separate_training_test_data(ratings_by_reviewer, test_portion):
    """
    Separates the ratings data for each reviewer into training and test sets.

    Args:
        ratings_by_reviewer (dict): A dictionary containing the ratings data for each reviewer. The reviewer ID should be the key, and the ratings should be stored as a nested dictionary.
        test_portion (float): The portion of data to be allocated for the test set (between 0 and 1).

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the training data, and the second dictionary contains the test data.
    """
    ratings_by_reviewer_training = {}
    ratings_by_reviewer_test = {}
    
    for reviewer in ratings_by_reviewer:
        music_reviewed_ids = list(ratings_by_reviewer[reviewer].keys())
        np.random.shuffle(music_reviewed_ids)  # Shuffle the list of music reviewed IDs
        
        split_index = int(len(music_reviewed_ids) * (1 - test_portion))
        training_ids = music_reviewed_ids[:split_index]
        test_ids = music_reviewed_ids[split_index:]
        
        ratings_by_reviewer_training[reviewer] = {music_id: ratings_by_reviewer[reviewer][music_id] for music_id in training_ids}
        ratings_by_reviewer_test[reviewer] = {music_id: ratings_by_reviewer[reviewer][music_id] for music_id in test_ids}
    
    return ratings_by_reviewer_training, ratings_by_reviewer_test

def get_item_features_and_rewards_for_user(n_features, ratings_for_selected_user, features_by_musics_dict):
  """
    Retrieves the item features and rewards for a selected user.

    Args:
        n_features (int): The number of features in the item vectors.
        ratings_for_selected_user (dict): A dictionary containing the ratings for the selected user. The music ID should be the key, and the rating should be the value.
        features_by_musics_dict (dict): A dictionary containing the feature vectors for each music. The music ID should be the key, and the feature vector should be a numpy array.

    Returns:
        tuple: A tuple containing the item features, item rewards, and the order of items.
    """
  n_arms = len(ratings_for_selected_user.keys())
  item_features = np.zeros((n_features, n_arms))
  item_rewards = np.zeros(n_arms)
  items_order = []

  index = 0
  for music_id in ratings_for_selected_user:
    feature_vector = features_by_musics_dict[music_id]
    item_features[:,index] = feature_vector
    item_rewards[index] = ratings_for_selected_user[music_id]
    items_order.append(music_id)
    index += 1

  return item_features, item_rewards, items_order
