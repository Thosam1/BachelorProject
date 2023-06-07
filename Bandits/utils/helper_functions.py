"""
    This document contains useful helper functions that are used and required for multiple notebooks to work.
"""
import numpy as np

def probability_distribution_by_gaps(n_arms, gaps):
    """
    Generates probability distributions for each sub-optimality gap.

    Args:
        n_arms (int): The number of arms in the multi-armed bandit.
        gaps (list): A list of sub-optimality gaps which represents the gap between the best arm and the second best arm.

    Returns:
        list: A list of probability distributions for each sub-optimality gap.
    """
    array_probs = []

    # Iterate over each sub-optimality gap
    for gap in gaps:
        # probability of success for the second best arm as a random number between 0 and 0.5
        p_second_best = np.random.rand() * (1 - gap)

        # probability of success for the best arm
        p_best = p_second_best + gap

        # calculate the other arm probabilities as random numbers between 0 and p_second_best
        probs = np.random.rand(n_arms - 2) * p_second_best

        # add the best and second best arm probabilities
        probs = np.append(probs, [p_best, p_second_best])

        # shuffle the probabilities and add them to the list
        np.random.shuffle(probs)
        array_probs.append(probs)

    return array_probs
