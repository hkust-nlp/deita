import os
from typing import List
from deita.pipeline.embed_pipeline import EmbedPipeline
from deita.selection.embedder import CLM_Prober
import logging
import pandas

logger = logging.getLogger(__name__)

class ProbePipeline(EmbedPipeline):
    
    def __init__(self, name: str, data_path: str, **kwargs) -> None:
        
        super().__init__(name, data_path, **kwargs)
        self.embedder = CLM_Prober(**kwargs)
            
        
        