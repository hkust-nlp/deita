import json
from typing import List

INSTRUCTION_ONLY_TYPE = ["complexity"]
INSTRUCTION_RESPONSE_TYPE = ["quality"]

def sort_key_split(sort_key: str) -> List:
    """
    Split sort_key into a list of sort keys.
    
    sort_key: str - sort key to split.
    """
    if "," in sort_key:
        return sort_key.split(",")
    elif "." in sort_key:
        return sort_key.split(".")
    elif "+" in sort_key:
        return sort_key.split("+")
    else:
        raise ValueError("sort_key must be a string with delimiter ',' or '.' or '+'.")

def load_data(self, data_path: str) -> None:
    
    """
    Load data from data_path.
    
    data_path: str - path to json data file.
    """
    
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
    
    return data