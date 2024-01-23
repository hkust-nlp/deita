import json

INSTRUCTION_ONLY_TYPE = ["complexity"]
INSTRUCTION_RESPONSE_TYPE = ["quality"]

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