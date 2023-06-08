"""
    This file contains functions for calculating different metrics and distances between vectors.
"""

import numpy as np

def calculate_mse(vector1, vector2):
    """
    Calculates the Mean Squared Error (MSE) between two vectors.

    Args:
        vector1 (numpy.ndarray): The first vector.
        vector2 (numpy.ndarray): The second vector.

    Returns:
        float: The MSE between the two vectors.
    """
    # Check if the vectors have the same shape
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape.")

    # Calculate the squared difference between the vectors
    squared_diff = np.square(vector1 - vector2)

    # Calculate the mean of the squared differences
    mse = np.mean(squared_diff)

    return mse

def calculate_absolute_difference(vector1, vector2):
    """
    Calculates the absolute difference between two vectors.

    Args:
        vector1 (numpy.ndarray): The first vector.
        vector2 (numpy.ndarray): The second vector.

    Returns:
        float: The total absolute difference between the two vectors.
    """
    # Check if the vectors have the same shape
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape.")

    # Calculate the absolute difference between the vectors
    absolute_diff = np.abs(np.subtract(vector1, vector2))

    # Calculate the total absolute difference
    total_diff = np.sum(absolute_diff)

    return total_diff

def calculate_euclidean_distance(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors.

    Args:
        vector1 (numpy.ndarray): The first vector.
        vector2 (numpy.ndarray): The second vector.

    Returns:
        float: The Euclidean distance between the two vectors.
    """
    # Check if the vectors have the same shape
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape.")

    # Calculate the squared difference between the vectors
    squared_diff = np.square(vector1 - vector2)

    # Calculate the sum of the squared differences
    sum_squared_diff = np.sum(squared_diff)

    # Calculate the square root of the sum
    euclidean_distance = np.sqrt(sum_squared_diff)

    return euclidean_distance