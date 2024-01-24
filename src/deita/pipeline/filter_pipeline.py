import os
import json
import pandas as pd
from typing import List
from deita.pipeline.base import BasePipeline
from deita.pipeline.utils import sort_key_split
from deita.selection.filter import Combined_Filter
import logging
import pandas
import numpy as np

logger = logging.getLogger(__name__)

class FilterPipeline(BasePipeline):
    
    def __init__(self, name: str, data_path: str, **kwargs) -> None:
        
        self.name = name
        self.data_path = data_path
        self.other_data_path = kwargs.get("other_data_path")
        self.is_compression = kwargs.get("is_compression", False)
        
        self.data_format = "sharegpt"   # only support sharegpt for now        
        self.output_path = kwargs.get("output_path")
        self.sort_key = kwargs.get("sort_key")
        self.sort_key = sort_key_split(self.sort_key)
        kwargs["sort_key"] = self.sort_key
        
        if not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
        self.filter = Combined_Filter(**kwargs)
            
    def _load_other_data(self, other_data_path: str) -> None:
        """
            Load Embedding Data
        """
        
        if self.is_compression:
            embedding_data = pd.read_pickle(other_data_path, "zip")
        else:
            embedding_data = pd.read_pickle(other_data_path)
            
        return embedding_data

    def _preprocess(self, json_data, other_data) -> List:
        
        """
            json_data: List - data to be filtered
            other_data: pd.DataFrame - embedding data
        """

        if isinstance(other_data, np.ndarray):
            df_data = pd.DataFrame([{"embedding": other_data[i]} for i in range(other_data.shape[0])])
        elif isinstance(other_data, pd.DataFrame):
            df_data = other_data
        else:
            raise ValueError("other_data must be either np.array or pd.DataFrame")

        if "idx" not in df_data.columns:
            df_data["idx"] = df_data.index
        
        df_json = pd.DataFrame(json_data)
        
        for sk in self.sort_key:
            df_data[sk] = df_json[sk].tolist()
            
        return df_data
        
    def _forward(self, preprocessed_data) -> List:
        
        selected_data = self.filter.filter(preprocessed_data)
        selected_data_indices = selected_data["idx"].tolist()
        
        logger.info(f"Selected Data Number: {len(selected_data_indices)}")
        logger.info("Finished Combined Selection")
        
        return selected_data_indices
    
    def _save_data(self, json_data: List, results: List) -> None:
        
        selected_data = []
        
        for idx in results:
            selected_data.append(json_data[idx])
        
        with open(self.output_path, "w") as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)
            
        
        