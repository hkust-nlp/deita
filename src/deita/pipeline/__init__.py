from .embed_pipeline import EmbedPipeline
from .score_pipeline import ScorePipeline
from .base import PipelineRegistry
from typing import Any, Dict, List, Optional, Union, Callable

PipelineRegistry.register("score_pipeline", ScorePipeline)
PipelineRegistry.register("embed_pipeline", EmbedPipeline)

class Pipeline:
    
    def __new__(cls, name, **kwargs) -> Callable:
        
        PipelineClass = PipelineRegistry.get_pipeline(name)
        return PipelineClass(name, **kwargs)