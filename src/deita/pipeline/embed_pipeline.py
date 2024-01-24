import os
from typing import List
from deita.pipeline.base import BasePipeline
from deita.selection.embedder import CLM_Embedder
import logging
import pandas

logger = logging.getLogger(__name__)

class EmbedPipeline(BasePipeline):
    
    def __init__(self, name: str, data_path: str, **kwargs) -> None:
        
        self.name = name
        self.data_path = data_path
        self.is_compression = kwargs.get("is_compression", False)
        
        self.data_format = "sharegpt"   # only support sharegpt for now        
        self.output_path = kwargs.get("output_path")
        
        if not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
        self.embedder = CLM_Embedder(**kwargs)
            
    def _preprocess(self, json_data, other_data) -> List:
        return json_data
    
    def _forward(self, preprocessed_data) -> List:
        
        all_embeddings_list = self.embedder.encode_samples(preprocessed_data)
        
        logger.info(f"{len(all_embeddings_list)}")
        logger.info("Finished embedding")
        
        return all_embeddings_list
    
    def _save_data(self, json_data: List, results: List) -> None:
        
        # We use dataframe to save the results
        df = pandas.DataFrame(results)
        
        if self.embedder.accelerator.is_main_process:
            df.sort_values(by = "idx", inplace = True)
            df.reset_index(drop = True, inplace = True)
            
            if not self.is_compression:
                df.to_pickle(self.output_path)
                logger.info(f"Saved pickle to {self.output_path}")
            else:
                df.to_pickle(self.output_path, "zip")
                logger.info(f"Saved pickle to {self.output_path} with zip compression")
            
        
        