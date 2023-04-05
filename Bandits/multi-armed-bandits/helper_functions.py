import numpy as np
import matplotlib.pyplot as plt

# Helper functions that are used for multi-armed bandits


def bar_plot(x_values, y_values, x_label="", y_label="", title=""):
    # Create bar plot
    plt.bar(x_values, y_values)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return plt


def bar_plot_explore_then_exploit(x_values, y_values, x_label="", y_label="", title="", true_probs=None):
    if true_probs is None:
        true_probs = []
    plt = bar_plot(x_values, y_values, x_label, y_label, title)

    for i, p in enumerate(true_probs):
        plt.axhline(y=p, color='red', linestyle='--')
        plt.text(i, p, f'{p:.0%}', ha='center', va='bottom')

    # Set y-axis limits to 0.0 and 1.0
    plt.ylim(0.0, 1.0)

    return plt


def line_plot(y_values, x_values=None, x_label="", y_label="", title=""):
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
    optimal_arm = np.argmax(p)
    sub_optimal_arm_chosen = np.cumsum((actions != optimal_arm).astype(int))

    plt = line_plot(y_values=sub_optimal_arm_chosen, x_label="Time",
                    y_label="Total nb of times sub-optimal arm was chosen", title="# times sub-optimal arm was chosen")
    return plt


def money_stats(initial_money, price_per_play, n_trials, reward, rewards):
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
    diff_per_play = np.array(rewards).reshape((len(rewards), 1)) - np.ones((len(rewards), 1)) * price_per_play
    plt = line_plot(y_values=np.cumsum(diff_per_play), x_label="Time", y_label="Money difference",
                    title="Money gain/loss over time")

    return plt


def money_plot_explore_then_exploit(rewards, price_per_play, changes):
    plt = money_plot(rewards, price_per_play)

    plt = add_vertical_lines(plt, changes)

    return plt


def add_vertical_lines(plt, changes):
    percent_10_below = plt.ylim()[0] - 0.1 * plt.ylim()[1]

    # Add vertical dashed lines
    for step_arm in changes:
        plt.axvline(x=step_arm[0], linestyle='--', color='red')
        plt.text(step_arm[0], percent_10_below, 'Arm ' + str(step_arm[1]), rotation=90, va='top')

    return plt


def regret_plot_explore_then_exploit(y_values, x_values=None, title="", changes=None):
    if changes is None:
        changes = []
    plt = line_plot(y_values, x_values, "Time", "Regret", title)
    plt = add_vertical_lines(plt, changes)
    return plt


def show_plot(plt):
    plt.show()


# Define the regret function
def regret(actions, p):
    actions_array = actions.reshape(len(actions), 1).astype(int)
    return np.cumsum(np.max(p) - p[actions_array])


def plot_estimate_interval(n_arms, wins, losses, c):
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

    # Plot bar plot with estimate interval
    x = np.arange(n_arms)
    fig, ax = plt.subplots()
    ax.bar(x, p, align='center', alpha=0.5)
    ax.vlines(x, lower_bounds, upper_bounds, colors='r', linewidth=2)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Probability Estimate')
    ax.set_title('Estimate Interval')
    plt.show()


def empirical_probabilities_with_eliminations(results_all_arms_end_of_explore, end_of_explore_total, eliminations):
    empirical_probabilities = end_of_explore_total / results_all_arms_end_of_explore.shape[1]
    for arm_step in eliminations:
        empirical_probabilities[arm_step[0]] = end_of_explore_total[arm_step[0]] / arm_step[2]
    return empirical_probabilities
