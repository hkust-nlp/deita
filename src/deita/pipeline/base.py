import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)

class BasePipeline:
    
    def __init__(self, name: str, data_path: str, **kwargs) -> None:
        
        self.name = name
        self.data_path = data_path
    
    def _load_data(self, data_path: str) -> None:
        
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
    
    def _load_other_data(self, other_data_path: str) -> None:
        raise NotImplementedError
        
    def _save_data(self, data_path: str, data_format: str) -> None:
        raise NotImplementedError
    
    def _preprocess(self, json_data, other_data) -> None:
        raise NotImplementedError
    
    def _forward(self, preprocessed_data) -> None:
        raise NotImplementedError
    
    def run(self) -> None:
        
        json_data = self._load_data(self.data_path)
        
        other_data = None
        if hasattr(self, "other_data_path"):
            other_data = self._load_other_data(self.other_data_path)
        
        preprocessed_data = self._preprocess(json_data, other_data)
        results = self._forward(preprocessed_data)
        self._save_data(json_data, results)
        logger.info(f"Pipeline {self.name} run complete.")
        
class PipelineRegistry:
    
    registry = {}
    
    @classmethod
    def register(cls, name: str, pipline_class: Callable):
        
        if name in cls.registry:
            raise ValueError(f"Pipeline {name} already registered.")
        cls.registry[name] = pipline_class
    
    @classmethod
    def get_pipeline(cls, name: str):
        
        if name not in cls.registry:
            raise ValueError(f"Pipeline {name} not registered.")
        return cls.registry[name]
    
