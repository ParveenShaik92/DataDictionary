import json
from typing import List

def read_json_file(filepath: str) -> List[dict]:
    """
    Reads a JSON file containing a list of records (list of dicts).
    Returns the data as a list of dictionaries.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of records (list of dictionaries).")
        return data

def get_json_headers(data: List[dict]) -> List[str]:
    """
    Returns the list of column headers (keys from the first record).
    """
    if not data:
        return []
    return list(data[0].keys())
