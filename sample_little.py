import sys
import json
import random

filepath = sys.argv[1]
outputpath = sys.argv[2]

data = json.load(open(filepath, "r"))

sampled_data = random.sample(data, 100)
with open(outputpath, "w") as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)