import csv
import os

def read_csv_file(filepath: str, has_header: bool = True) -> list[list[str]]:
    """
    Reads data from a CSV file and returns all rows as a list of lists.

    Args:
        filepath (str): The path to the CSV file.
        has_header (bool): Set to True if the CSV file has a header row (default: True).
                           If True, the header row will be excluded from the returned data.

    Returns:
        list[list[str]]: A list where each inner list represents a row of data
                         from the CSV file. Returns an empty list if the file
                         is empty or an error occurs.
    """
    data_rows = []
    if not os.path.exists(filepath):
        print(f"Error: CSV file not found at '{filepath}'")
        return []

    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            if has_header:
                # Skip the header row if present
                next(reader, None)

            for row in reader:
                data_rows.append(row)
        print(f"Successfully read data from '{filepath}'. Total rows: {len(data_rows)}")
        return data_rows
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV file '{filepath}': {e}")
        return []
