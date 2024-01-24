import torch
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class IterativeFilter(object):
    def __init__(self, **kwargs):
        
        self.threshold = kwargs.get('threshold')
        self.data_size = kwargs.get('data_size')
        self.sort_key = kwargs.get('sort_key')
        self.chunk_size = kwargs.get('chunk_size')
        self.batch_size = kwargs.get('batch_size', 1)
        self.normalize_emb = kwargs.get('normalize_emb', True)
        self.distance_metric = kwargs.get('distance_metric', 'cosine')
        self.embedding_field = kwargs.get('embedding_field', "embedding")
        
        self.device = kwargs.get('device') if kwargs.get('device') == 'cpu' else f"cuda:{kwargs.get('device')}"
    
    def compute_distance(self, matrix, matrix_2):
        
        """
            Compute cosine distance using pytorch
        """
        
        if self.normalize_emb:
            matrix = matrix / matrix.norm(dim=1)[:, None]
            matrix_2 = matrix_2 / matrix_2.norm(dim=1)[:, None]

        if self.distance_metric == 'cosine':
            matrix_norm = matrix / matrix.norm(dim=1)[:, None]
            matrix_2_norm = matrix_2 / matrix_2.norm(dim=1)[:, None]
            return torch.mm(matrix_norm, matrix_2_norm.t())
        elif self.distance_metric == 'manhattan':
            return torch.cdist(matrix[None], matrix_2[None], p = 1).squeeze(0)
        else:
            raise ValueError("Metric not supported. Only support cosine and manhattan")
    
    def _sort(self, df):

        raise NotImplementedError
    
    def distance_chunk_by_chunk(self, existing_emb, cur_emb):
        
        distance_placeholder = torch.zeros((cur_emb.size(0), existing_emb.shape[0]), dtype = torch.float32).to(self.device)

        for i in range(0, existing_emb.shape[0], self.chunk_size):
            
            chunk_embeddings = existing_emb[i: i + self.chunk_size]
            chunk_embeddings = torch.tensor(chunk_embeddings, dtype = torch.float32).to(self.device)
            
            if chunk_embeddings.ndim == 4:
                chunk_embeddings = chunk_embeddings.squeeze(1).squeeze(1)

            distance_matrix = self.compute_distance(cur_emb, chunk_embeddings)
            actual_chunk = distance_matrix.size(1)
            
            distance_placeholder[:, i: i + actual_chunk] = distance_matrix

        return distance_placeholder

    def filter(self, df):
        
        logger.info(f"Data number before filtering: #{len(df)}")
        
        df_sorted = self._sort(df)

        embeddings = df_sorted[self.embedding_field]
        embeddings = np.array(embeddings.values.tolist())
        
        filtered_indices = [0]

        start_cnt = 0
        for i in tqdm(range(1, embeddings.shape[0], self.batch_size), total = embeddings.shape[0] // self.batch_size):

            cur_emb = torch.tensor(embeddings[i:i+self.batch_size], dtype = torch.float32).to(self.device)
            
            if cur_emb.ndim == 4:
                cur_emb = cur_emb.squeeze(1).squeeze(1)

            if cur_emb.ndim == 1:
                cur_emb = cur_emb.unsqueeze(0)

            batch_idx = torch.range(i, i + cur_emb.size(0) - 1, dtype = torch.int64).to(self.device)
            
            existing_emb = embeddings[filtered_indices]

            if existing_emb.ndim == 1:
                existing_emb = existing_emb.unsqueeze(0)

            distance_existed = self.distance_chunk_by_chunk(existing_emb, cur_emb)
            distance_existed_bool = torch.any(distance_existed > self.threshold, dim = 1)
            
            distance_cur = self.distance_chunk_by_chunk(cur_emb, cur_emb)
            distance_cur = distance_cur.tril(-1)
            
            distance_cur_bool = torch.any(distance_cur > self.threshold, dim = 1)
            
            distance_bool = distance_existed_bool | distance_cur_bool
            
            filtered_indices.extend(batch_idx[~distance_bool].tolist())

            if len(filtered_indices) - start_cnt > 1000:
                logger.info("Now data number: #{}".format(len(filtered_indices)))
                start_cnt = len(filtered_indices)

            if self.data_size > -1:
                if len(filtered_indices) >= self.data_size:
                    break
            
        df_filtered = df_sorted.iloc[filtered_indices]        
        logger.info(f"Data number after filtering: #{len(df_filtered)}")
        
        if self.data_size > -1:
            return df_filtered[:self.data_size]
        else:
            return df_filtered
        
        