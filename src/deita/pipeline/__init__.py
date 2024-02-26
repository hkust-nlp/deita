from .embed_pipeline import EmbedPipeline
from .score_pipeline import ScorePipeline
from .filter_pipeline import FilterPipeline
from .probe_pipeline import ProbePipeline
from .base import PipelineRegistry
from typing import Callable

PipelineRegistry.register("score_pipeline", ScorePipeline)
PipelineRegistry.register("embed_pipeline", EmbedPipeline)
PipelineRegistry.register("filter_pipeline", FilterPipeline)
PipelineRegistry.register("probe_pipeline", ProbePipeline)

class Pipeline:
    
    def __new__(cls, name, **kwargs) -> Callable:
        
        PipelineClass = PipelineRegistry.get_pipeline(name)
        return PipelineClass(name, **kwargs)