import os
import json
from deita.pipeline.base import BasePipeline
from deita.selection.scorer import Llama_Scorer, Mistral_Scorer
from typing import Any, Dict, List, Optional, Tuple, Union
from deita.pipeline.utils import load_data
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ScorePipeline(BasePipeline):
    
    def __init__(self, name: str, data_path: str, **kwargs) -> None:
        
        self.name = name
        self.data_path = data_path
        
        self.data_format = "sharegpt"   # only support sharegpt for now
        
        scorer = kwargs.get("scorer")
        is_vllm = kwargs.get("is_vllm")
        scorer_name_or_path = kwargs.get("scorer_name_or_path")
        self.score_type = kwargs.get("score_type")
        
        if scorer == "llama":
            self.model = Llama_Scorer(scorer_name_or_path, is_vllm)
        elif scorer == "mistral":
            self.model = Mistral_Scorer(scorer_name_or_path, is_vllm)
        
        self.output_path = kwargs.get("output_path")
        
        if not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
    def _load_sharegpt(self, data: str) -> List:
        
        preprocessed_data = []
        
        for sample_id, item in enumerate(data):
            
            preprocessed_item = []
            
            for idx in range(len(item["conversations"])):
                
                if idx % 2 != 0:
                    continue
                
                if idx != len(item["conversations"]) - 1:
                    preprocessed_item.append({"instruction": item["conversations"][idx]["value"], "response": item["conversations"][idx+1]["value"]})
                else:
                    preprocessed_item.append({"instruction": item["conversations"][idx]["value"], "response": ""})
                    
            preprocessed_data.append({"conversations": preprocessed_item, "n_conv": len(preprocessed_item)})
            
        return preprocessed_data
    
    def _inject_sharegpt(self, json_data: List, results: List) -> None:
        
        for sample_id in range(len(json_data)):
            
            json_data[sample_id][f"{self.score_type}_scores"] = []
                
            for item in results[sample_id]["conversations"]:
                json_data[sample_id][f"{self.score_type}_scores"].append(float(item[f"{self.score_type}_score"]))
        
        
    def _preprocess(self, json_data, other_data) -> List:
        
        if self.data_format == "sharegpt":
            preprocessed_data = self._load_sharegpt(json_data)
        else:
            raise ValueError(f"Data format {self.data_format} not supported.")
        
        return preprocessed_data
    
    def _forward(self, preprocessed_data) -> List:
        
        for convs in tqdm(preprocessed_data, total = len(preprocessed_data)):
            
            for conv in convs["conversations"]:
                
                if self.score_type.lower() == "complexity":
                    score = self.model.infer_complexity(conv["instruction"])
                elif self.score_type.lower() == "quality":
                    score = self.model.infer_quality(conv["instruction"], conv["response"])
                else:
                    raise ValueError(f"Score type {self.score_type} not supported.")
                
                conv[f"{self.score_type}_score"] = score
                
        return preprocessed_data
    
    def _save_data(self, json_data: List, results: List) -> None:
        
        if self.data_format == "sharegpt":
            self._inject_sharegpt(json_data, results)
        else:
            raise ValueError(f"Data format {self.data_format} not supported.")
        
        with open(self.output_path, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved results to {self.output_path}.")