import numpy as np
import matplotlib.pyplot as plt

def plot_ratings_for_user(user_id, dictionary):
    """
    Plots the count of ratings for a specific user.

    Args:
        user_id (str): The ID of the user.
        dictionary (dict): A dictionary containing the ratings data. The user ID should be the key and the ratings should be stored as values.

    Returns:
        None
    """
    ratings_count = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }
    ratings_list = dictionary[user_id].values()
    for rating in ratings_list:
        if len(rating) != 0:
            ratings_count[int(float(rating))] += 1

    ratings = sorted(ratings_count.items())

    x = [rating[0] for rating in ratings]
    y = [rating[1] for rating in ratings]

    plt.bar(x, y)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Count of Ratings for user ' + user_id)
    plt.show()

def plot_dict_list(list_of_dict, x_label, y_label, title, legends):
    """
    Plots multiple dictionaries as lines on a single graph.

    Args:
        list_of_dict (list): A list of dictionaries to be plotted. Each dictionary represents a line on the graph.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the graph.
        legends (list): A list of legends for the lines.

    Returns:
        None
    """
    x_values = list(list_of_dict[0].keys())

    fig, ax = plt.subplots()

    for i, dictionary in enumerate(list_of_dict):
        y_values = [dictionary[key] for key in x_values]
        ax.plot(x_values, y_values, marker='o', label=legends[i])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()
