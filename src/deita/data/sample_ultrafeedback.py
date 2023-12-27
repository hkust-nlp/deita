import os
import sys
import random
from datasets import load_dataset, Dataset

outputpath = sys.argv[1]
datanum = sys.argv[2]

random.seed(42)

if not os.path.exists(outputpath):
    data = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split = "train_prefs")
    data.to_json(outputpath)
    
sample_indices = random.sample(range(len(data)), int(datanum))

sampled_data = data.select(sample_indices)
sampled_data.to_json(outputpath)