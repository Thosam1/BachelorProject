import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_regrets(list_of_regrets, list_of_labels):
    """
    Plots the multiple lines of cumulative regrets over time.

    Args:
        list_of_regrets (list): A list of numpy arrays, each containing cumulative regrets over time.
        list_of_labels (list): A list of labels corresponding to each set of cumulative regrets.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    for i, cumulative_regrets in enumerate(list_of_regrets):
        ax.plot(cumulative_regrets, label=list_of_labels[i])

    ax.set(xlabel='Time steps', ylabel='Cumulative regret', title='Cumulative regret over time')
    ax.grid()
    ax.legend()
    plt.show()