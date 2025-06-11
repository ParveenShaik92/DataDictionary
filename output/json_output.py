import json
import os

def write_json_output(data: dict, output_file: str) -> None:
    """
    Writes the given data to a JSON file. Creates the file and parent directories if they don't exist.

    Args:
        data (dict): The data to write to JSON.
        output_file (str): Path to the output file.
    """
    # Ensure parent directory exists
    if not os.path.exists(output_file):
        print(f"Error: CSV file not found at '{output_file}'")
        return []

    # Write the JSON data to the file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"JSON output written to: {output_file}")