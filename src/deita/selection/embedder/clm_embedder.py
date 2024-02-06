import torch
from functools import partial

from tqdm import tqdm
import torch
from datasets import Dataset
import torch.distributed as dist
from deita.selection.embedder.base import Embedder
from deita.selection.embedder.utils import DataCollatorForSupervisedDataset, preprocess

import logging

logger = logging.getLogger(__name__)

class CLM_Embedder(Embedder):

    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
    
    def encode_samples(self, data):

        conversations = [item["conversations"] for item in data]
        dataset_buf, data_size = self.create_databuffer(conversations, sort_by_length = True)
        raw_dataset = Dataset.from_list(dataset_buf)

        preprocess_func = partial(preprocess, 
                                conv_template = self.conv_template,
                                only_answer = self.only_answer,
                                max_length = self.max_length,
                                tokenizer = self.tokenizer)
        
        with self.accelerator.main_process_first():
          tokenized_datasets = raw_dataset.map(
              preprocess_func,
              batched = True,
              num_proc = 32,
              remove_columns = ["conversations", "specific_length"],
              desc = "Tokenizing and reformatting instruction data"
          )  
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer = self.tokenizer)
        dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size = self.batch_size_per_device, collate_fn = data_collator)
        
        model, dataloader = self.accelerator.prepare(self.model, dataloader)
        
        all_embeddings_list = []
        
        total_samples = len(tokenized_datasets)
        total_batches = len(dataloader)
        last_batch_size = total_samples % self.minibatch_size if total_samples % self.minibatch_size != 0 else self.minibatch_size
        
        for b_idx, batch in enumerate(tqdm(dataloader, total = len(tokenized_datasets) // self.minibatch_size, disable = not self.accelerator.is_local_main_process)):
            
            model.eval()

            batch_idx = batch["idx"]
            attention_mask = batch["attention_mask"]

            outputs = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], output_hidden_states = True)
            
            seq_len = attention_mask.sum(1, keepdim = True)
            
            if self.tokenizer.padding_side == "right":
                last_hidden_state = outputs.hidden_states[-1][torch.arange(seq_len.size(0))[:, None], seq_len - 1]
            elif self.tokenizer.padding_side == "left":    
                last_hidden_state = outputs.hidden_states[-1][:, -1]
            else:
                raise ValueError("Invalid padding strategy")
            
            sample_idx = batch_idx.tolist()
            sample_dict = [{"embedding": lst_hs, "idx": s_id} for lst_hs, s_id in zip(last_hidden_state.tolist(), sample_idx)]
            
            if(self.world_size > 1):
                all_process_embeddings = [[] for _ in range(self.world_size)]
                dist.gather_object(sample_dict, all_process_embeddings if dist.get_rank() == 0 else None, dst=0)
            else:
                all_process_embeddings = [sample_dict]
            
            if self.accelerator.is_local_main_process:
                if b_idx == total_batches - 1:
                    for process_list in all_process_embeddings[:last_batch_size]:
                        all_embeddings_list.extend(process_list)
                else:
                    for process_list in all_process_embeddings:
                        all_embeddings_list.extend(process_list)   
        
        return all_embeddings_list