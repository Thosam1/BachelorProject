import numpy as np
import matplotlib.pyplot as plt
import csv

# Read the given csv as a dict of dict
def read_csv_as_dict_of_dict(filename):
    result = {}
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Assuming the first column as the primary key
            key = row.pop(reader.fieldnames[0])
            filtered_row = {k: v for k, v in row.items() if v != ''}
            if filtered_row:
                result[key] = filtered_row
    return result


# Read the given csv as a dict of np arrays
def read_csv_as_dict_of_arrays(filename):
    result = {}
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.pop(reader.fieldnames[0])
            values = np.array([float(value) for value in row.values()])
            result[key] = values
    return result

# Removes a column in a csv file by the given column index
def remove_csv_column_by_index(input_file_path, output_file_path, column_to_remove):
    input_file_path = '../dimension_reduction/reduced_data_PCA_10D.csv'
    output_file_path = '../dimension_reduction/reduced_data_PCA_10D_.csv'
    column_to_remove = column_to_remove
    # Read the input CSV file and remove the specified column
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)
        rows = [row[:column_to_remove] + row[column_to_remove + 1:] for row in reader]

    # Write the modified data to the output CSV file
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Modified CSV file exported successfully.")

# Appends a column filled with 'value'
def append_column_in_csv(input_file_path, output_file_path, value):

    # Read the input CSV file and modify the rows
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)
        rows = [row + [str(value)] for row in reader]

    # Write the modified data to the output CSV file
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Modified CSV file exported successfully.")


def plot_ratings_for_user(user_id, dictionary):
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