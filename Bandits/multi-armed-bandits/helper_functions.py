import numpy as np
import matplotlib.pyplot as plt


# Helper functions that are used for multi-armed bandits


def bar_plot(x_values, y_values, x_label="", y_label="", title=""):
    '''
    Create a bar plot with the given data.

    Args:
        x_values (list): The values for the x-axis.
        y_values (list): The values for the y-axis.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.

    Returns:
        matplotlib plot: The bar plot.
    '''
    # Create bar plot
    plt.bar(x_values, y_values)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return plt


def bar_plot_explore_then_exploit(x_values, y_values, x_label="", y_label="", title="", true_probs=None):
    '''
    Create a bar plot with the given data, and add additional information to it.

    Args:
        x_values (list): The values for the x-axis.
        y_values (list): The values for the y-axis.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        true_probs (list): A list of true probabilities.

    Returns:
        matplotlib plot: The bar plot with additional information added to it.
    '''

    if true_probs is None:
        true_probs = []

    # Create bar plot
    plt = bar_plot(x_values, y_values, x_label, y_label, title)

    # Add true probabilities to the plot
    for i, p in enumerate(true_probs):
        plt.axhline(y=p, color='red', linestyle='--')
        plt.text(i, p, f'{p:.0%}', ha='center', va='bottom')

    # Set y-axis limits to 0.0 and 1.0
    plt.ylim(0.0, 1.0)

    return plt


def line_plot(y_values, x_values=None, x_label="", y_label="", title=""):
    """
    Create a line plot for given y values against corresponding x values, with optional x label, y label, and title.

    Args:
        y_values (list): list or array of y values
        x_values (list, optional): list or array of x values; if not specified, range of length of y_values will be used
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the plot

    Returns:
        matplotlib plot: The bar plot with additional information added to it.
    """

    # If x_values is not specified, use range of length of y_values
    if x_values is None:
        x_values = range(len(y_values))

    # Create line plot
    plt.plot(x_values, y_values)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return plt


def multiple_lines_plot(x_values, y_values_list, labels=None, x_label="", y_label="", title=""):
    """
    Create a plot with multiple lines, each corresponding to a list of y values, with optional x values, labels for each line, x label, y label, and title.

    Args:
        x_values (list): list or array of x values
        y_values_list (list): list of lists or arrays of y values; each list corresponds to a line in the plot
        labels (optional): list of labels for each line; if not specified, no legend will be shown
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the plot

    Returns:
        None (the plot is shown using matplotlib.pyplot.show())
    """
    # Create multiple lines plot
    if labels is None:
        labels = []
    for y_values, l in zip(y_values_list, labels):
        plt.plot(x_values, y_values, label='l')
        plt.legend(labels)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show the plot
    plt.show()


def sub_optimal_arm_chosen_plot(p, actions):
    """
    Plot the number of times the sub-optimal arm was chosen over time.

    Args:
        p (list):  the probabilities of success for each arm.
        actions (list): the actions taken over time, where each action is the index of the arm chosen.

    Returns:
        matplotlib plot: The plot for the total number of times the suboptimal arm was chosen over time.
    """
    optimal_arm = np.argmax(p)
    sub_optimal_arm_chosen = np.cumsum((actions != optimal_arm).astype(int))

    # Create line plot of the total number of times the sub-optimal arm was chosen over time
    plt = line_plot(y_values=sub_optimal_arm_chosen, x_label="Time",
                    y_label="Total nb of times sub-optimal arm was chosen", title="# times sub-optimal arm was chosen")

    return plt


def money_stats(initial_money, price_per_play, n_trials, reward, rewards):
    """
    Calculates and prints various statistics related to a money game simulation.

    Args:
        initial_money (float): the initial amount of money the player starts with
        price_per_play (float): the cost to play the game once
        n_trials (int): the number of times the player plays the game
        reward (float): the reward for winning one pull
        rewards (numpy.ndarray): an array of rewards obtained in each trial

    Returns:
        None

    Prints:
        "Money initially : " followed by the initial amount of money
        "Money invested : " followed by the total money invested to play the game
        "Money gained : " followed by the total money gained from playing the game
        "Money difference : " followed by the difference between money gained and invested
        "Money left : " followed by the final amount of money left after playing the game
    """
    money_invested = n_trials * price_per_play
    money_gained = np.sum(rewards) * reward

    money_diff = money_gained - money_invested
    money_left = initial_money + money_diff

    print("Money initially : " + str(initial_money))
    print("Money invested : " + str(money_invested))
    print("Money gained : " + str(money_gained))
    print("Money difference : " + str(money_diff))
    print("Money left : " + str(money_left))


def money_plot(rewards, price_per_play):
    """
    Creates a line plot of the cumulative difference between the total rewards earned and the money invested,
    based on the price per play and the rewards obtained over time.

    Args:
        rewards (list): list or array of reward values obtained at each trial
        price_per_play (float): price of a single play

    Returns:
        matplotlib.pyplot object: the line plot object
    """
    diff_per_play = np.array(rewards).reshape((len(rewards), 1)) - np.ones((len(rewards), 1)) * price_per_play
    plt = line_plot(y_values=np.cumsum(diff_per_play), x_label="Time", y_label="Money difference",
                    title="Money gain/loss over time")

    return plt


def money_plot_explore_then_exploit(rewards, price_per_play, changes):
    """
    Plot the money difference over time with vertical lines marking changes of arms.

    Args:
        rewards (list): list of rewards received for each play
        price_per_play (float): cost per play
        changes (list of tuples): list of indices indicating when an arm is switched

    Returns:
        plt: plot of money difference over time with vertical lines marking changes of arms
    """
    plt = money_plot(rewards, price_per_play)

    plt = add_vertical_lines(plt, changes)

    return plt


def add_vertical_lines(plt, changes):
    """
    Add vertical dashed lines to a plot, corresponding to arm changes in the plot at specific x values, with label
    for each arm.

    Args: plt: plot object from matplotlib.pyplot changes (list of tuples): each tuple contains two elements,
    the x value at which to add the vertical line, and a string describing the arm index

    Returns:
        plot object from matplotlib.pyplot with vertical lines and labels added
    """
    percent_10_below = plt.ylim()[0] - 0.1 * plt.ylim()[1]

    # Add vertical dashed lines
    for step_arm in changes:
        plt.axvline(x=step_arm[0], linestyle='--', color='red')
        plt.text(step_arm[0], percent_10_below, step_arm[1], rotation=90, va='top')

    return plt


def regret_plot_explore_then_exploit(y_values, x_values=None, title="", changes=None):
    """
    Create a plot of regret over time, with optional x values, title, and vertical lines marking changes.

    Args:
        y_values (list or array): list or array of y values
        x_values (optional): list or array of x values; if not specified, the indices of y_values are used as x values
        title (str): title for the plot
        changes (optional): list of tuples representing changes in the experiment; each tuple is of the form (time, arm); if not specified, no vertical lines are shown

    Returns:
        None (the plot is shown using matplotlib.pyplot.show())
    """
    if changes is None:
        changes = []
    plt = line_plot(y_values, x_values, "Time", "Regret", title)
    plt = add_vertical_lines(plt, changes)
    return plt


def show_plot(plt):
    """
    Displays a given plot.

    Args:
        plt (plot): plot to be displayed using matplotlib.pyplot.show()

    Returns:
        None
    """
    plt.show()


def regret(actions, p):
    """
    Calculate the cumulative regret over time, given a set of chosen actions and the true probability distribution.

    Args:
        actions (numpy.ndarray of int): 1D numpy array of chosen actions over time
        p (numpy.ndarray of float): 1D numpy array of true probabilities for each action

    Returns:
        array: 1D numpy array of cumulative regret over time
    """
    actions_array = actions.reshape(len(actions), 1).astype(int)
    return np.cumsum(np.max(p) - p[actions_array])


def plot_estimate_interval(n_arms, wins, losses, c):
    """
    Plots a bar plot with the estimate interval for each arm.

    Args:
        n_arms (int): Number of arms.
        wins (numpy.ndarray): Array containing the number of wins for each arm.
        losses (numpy.ndarray): Array containing the number of losses for each arm.
        c (float): Confidence level for the estimate interval (constant).

    Returns:
        None
    """
    # Calculate upper and lower confidence bounds for each arm
    p = np.zeros(n_arms)
    lower_bounds = np.zeros(n_arms)
    upper_bounds = np.zeros(n_arms)
    for i in range(n_arms):
        if wins[i] + losses[i] > 0:
            p[i] = wins[i] / (wins[i] + losses[i])
            lower_bounds[i] = p[i] - c / np.sqrt(wins[i] + losses[i])
            upper_bounds[i] = p[i] + c / np.sqrt(wins[i] + losses[i])
        else:
            p[i] = 0
            lower_bounds[i] = 0
            upper_bounds[i] = 0

    # Plot bar plot with an estimate interval
    x = np.arange(n_arms)
    fig, ax = plt.subplots()
    ax.bar(x, p, align='center', alpha=0.5)
    ax.vlines(x, lower_bounds, upper_bounds, colors='r', linewidth=2)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Probability Estimate')
    ax.set_title('Estimate Interval')
    plt.show()


def empirical_probabilities_with_eliminations(results_all_arms_end_of_explore, end_of_explore_total, eliminations):
    """
    Calculates the empirical probabilities of each arm given the results of an experiment that includes
    some eliminations.

    Args:
    results_all_arms_end_of_explore: A numpy array of shape (n_arms, n_trials) where each element is either a 0 or 1
                                     indicating whether the corresponding arm was successful in the corresponding trial.
    end_of_explore_total: A numpy array of shape (n_arms,) where each element is the total number of times the
                          corresponding arm was successful at the end of the exploration phase.
    eliminations: A list of tuples, where each tuple contains the index of the eliminated arm, the step at which it was
                  eliminated, and the total number of times it was played until that step.

    Returns:
    A numpy array of shape (n_arms,) containing the empirical probabilities of each arm.
    """
    empirical_probabilities = end_of_explore_total / results_all_arms_end_of_explore.shape[1]
    for arm_step in eliminations:
        empirical_probabilities[arm_step[0]] = end_of_explore_total[arm_step[0]] / arm_step[2]
    return empirical_probabilities

def plot_cumulative_regrets(list_of_regrets, list_of_labels):
    fig, ax = plt.subplots()
    for i, cumulative_regrets in enumerate(list_of_regrets):
        ax.plot(cumulative_regrets, label=list_of_labels[i])

    ax.set(xlabel='Time steps', ylabel='Cumulative regret', title='Cumulative regret over time')
    ax.grid()
    ax.legend()
    plt.show()
