"""
    This file contains functions thats read and manipulate CSV files.
"""

import csv
import numpy as np

def read_csv_as_dict_of_dict(filename):
    """
    Reads a CSV file and returns its content as a dictionary of dictionaries.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        dict: A dictionary of dictionaries representing the CSV content. The first column is assumed to be the primary key.
    """
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

def read_csv_as_dict_of_arrays(filename):
    """
    Reads a CSV file and returns its content as a dictionary of NumPy arrays.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        dict: A dictionary of NumPy arrays representing the CSV content. The first column is used as the dictionary key.
    """
    result = {}
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.pop(reader.fieldnames[0])
            values = np.array([float(value) for value in row.values()])
            result[key] = values
    return result

def remove_csv_column_by_index(input_file_path, output_file_path, column_to_remove):
    """
    Removes a column from a CSV file based on the given column index.

    Args:
        input_file_path (str): The path to the input CSV file.
        output_file_path (str): The path to the output CSV file.
        column_to_remove (int): The index of the column to be removed.

    Returns:
        None
    """
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

def append_column_in_csv(input_file_path, output_file_path, value):
    """
    Appends a column filled with the specified value to a CSV file.

    Args:
        input_file_path (str): The path to the input CSV file.
        output_file_path (str): The path to the output CSV file.
        value: The value to be filled in the appended column.

    Returns:
        None
    """
    # Read the input CSV file and modify the rows
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)
        rows = [row + [str(value)] for row in reader]

    # Write the modified data to the output CSV file
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Modified CSV file exported successfully.")