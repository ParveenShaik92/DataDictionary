import json
import os
import numpy as np
import pprint

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
    output = {}
    for item in data:
        for key, value in item.items():
            if key == "Missing_Column_values_Count":
                output[key] = value.to_dict()
            elif key == 'Stratified_Samples':
                if not value.empty:
                    value = value.replace({np.nan: None})
                    output[key] = value.to_dict(orient='records')
            elif key == "Column_Descriptions":
                output[key] = value.fillna("").astype(str).to_dict()
            else:
                output[key] = value
            

    # Write the JSON data to the file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)

    print(f"JSON output written to: {output_file}")

def write_console_output(data: dict, output_file: str) -> None:
    """
    Writes the given data to a JSON file. Creates the file and parent directories if they don't exist.

    Args:
        data (dict): The data to write to JSON.
        output_file (str): Path to the output file.
    """
    # Ensure parent directory exists
    if not os.path.exists(output_file):
        print("\n--- Processed Data (Console Output) ---")
        for row in data:
            for key, value in row.items():
                print(f"\n*** {key} ***")
                pprint.pprint(value)
        return []
    
    with open("output.txt", "w") as f:
        pp = pprint.PrettyPrinter(stream=f, indent=4)
        pp.pprint(data)
        f.write("\n--- End of Data ---\n")

    print(f"JSON output written to: {output_file}")