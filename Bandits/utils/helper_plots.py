"""
    This file contains useful helper functions that are related to plotting results.
"""

import matplotlib.pyplot as plt

def plot_multiple_lines(list_of_lines, list_of_labels, title, xlabel, ylabel):
    """
    Plots multiple lines on the same graph.

    Args:
        list_of_lines (list): A list of lines to be plotted.
        list_of_labels (list): A list of labels for each line.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None
    """
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot each line
    for i, line in enumerate(list_of_lines):
        ax.plot(line, label=list_of_labels[i])

    # Set the title, x-axis label, and y-axis label
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    # Display grid lines
    ax.grid()

    # Display legend
    ax.legend()

    # Show the plot
    plt.show()


def list_of_labels_from_values_list(list_of_values, prepend_text):
    """
    Generates a list of labels by appending a text prefix to each value in the given list.

    Args:
        list_of_values (list): The list of values.
        prepend_text (str): The text to be prepended to each value.

    Returns:
        list: A list of labels with the text prefix.
    """
    list_of_labels = []

    # Iterate over each value in the list
    for value in list_of_values:
        # Append the text prefix to the value and create the label
        label = prepend_text + str(value)

        # Add the label to the list
        list_of_labels.append(label)

    return list_of_labels

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
        None
    '''
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Create the bar plot
    ax.bar(x_values, y_values)

    # Set the x-axis label, y-axis label, and title
    ax.set(xlabel=x_label, ylabel=y_label, title=title)

    # Display grid lines
    ax.grid()

    # Show the plot
    plt.show()

