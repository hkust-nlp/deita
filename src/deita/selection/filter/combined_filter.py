import torch
import logging
import numpy as np
from deita.pipeline.utils import sort_key_split
from deita.selection.filter.base import IterativeFilter

logger = logging.getLogger(__name__)

class Combined_Filter(IterativeFilter):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
    def _sort(self, df):
        
        """        
            Sort dataframe by given method
        """

        if isinstance(self.sort_key, list):
            all_sort_keys = self.sort_key
        else:
            all_sort_keys = sort_key_split(self.sort_key)
            
        logger.info("Compute final score for each sample, consider {}".format("+".join(all_sort_keys)))
        for sk in all_sort_keys:
            df[sk] = df[sk].apply(np.array)
        
        df["final_score"] = df[all_sort_keys[0]]
        for i in range(1, len(all_sort_keys)):
            df["final_score"] = df["final_score"] * df[all_sort_keys[i]]
            
        df["final_score"] = df["final_score"].apply(lambda x: x.sum())
        df_sorted = df.sort_values(by = "final_score", ascending = False)
        
        return df_sorted
        
        