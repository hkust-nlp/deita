import torch
from functools import partial

from tqdm import tqdm
import torch
from datasets import Dataset
import torch.distributed as dist
from deita.selection.embedder.base import Embedder
from deita.selection.embedder.utils import preprocess
from transformers.data import DataCollatorForSeq2Seq

import logging

logger = logging.getLogger(__name__)

class CLM_Embedder(Embedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _probe(self, outputs, attention_mask, **kwargs):
        
        seq_len = attention_mask.sum(1, keepdim = True)
        
        if self.tokenizer.padding_side == "right":
            last_hidden_state = outputs.hidden_states[-1][torch.arange(seq_len.size(0))[:, None], seq_len - 1].squeeze(1)
        elif self.tokenizer.padding_side == "left":    
            last_hidden_state = outputs.hidden_states[-1][:, -1]
        else:
            raise ValueError("Invalid padding strategy")
        
        return last_hidden_state
    
    def _get_dataloader(self, data):
        
        conversations = [item["conversations"] for item in data]
        dataset_buf, data_size = self.create_databuffer(conversations, sort_by_length = True)
        raw_dataset = Dataset.from_list(dataset_buf)

        if hasattr(self, "mask_user"):
            mask_user = self.mask_user
        else:
            mask_user = False
            
        preprocess_func = partial(preprocess, 
                                conv_template = self.conv_template,
                                only_answer = self.only_answer,
                                max_length = self.max_length,
                                tokenizer = self.tokenizer,
                                mask_user = mask_user)

        tokenized_datasets = raw_dataset.map(
            preprocess_func,
            batched = False,
            num_proc = self.num_proc,
            remove_columns = ["conversations", "specific_length"],
            desc = "Tokenizing and reformatting instruction data"
        )  
        
        dataloader = torch.utils.data.DataLoader(tokenized_datasets, 
                                                 batch_size = self.minibatch_size, 
                                                 collate_fn=DataCollatorForSeq2Seq(
                                                     tokenizer=self.tokenizer,
                                                     model = self.model,
                                                     padding="longest"
                                                     )
                                                 )

        return dataloader, tokenized_datasets
    
    def encode_samples(self, data):

        dataloader, tokenized_datasets = self._get_dataloader(data)
        
        all_embeddings_list = []
        
        total_samples = len(tokenized_datasets)
        total_batches = len(dataloader)
        last_batch_size = total_samples % self.minibatch_size if total_samples % self.minibatch_size != 0 else self.minibatch_size
        
        for b_idx, batch in enumerate(tqdm(dataloader, total = len(tokenized_datasets) // self.minibatch_size, disable = not (self.local_rank == 0))):
            # print(f"local rank {self.local_rank} curent{b_idx * self.minibatch_size + (self.local_rank + 1) * self.batch_size_per_device} total {total_samples}")
            if b_idx * self.minibatch_size + (self.local_rank + 1) * self.batch_size_per_device > total_samples:                
                batch_idx = batch["idx"][:self.batch_size_per_device]
                input_ids = batch["input_ids"][:self.batch_size_per_device]
                attention_mask = batch["attention_mask"][:self.batch_size_per_device]
                labels = batch["labels"][:self.batch_size_per_device] if "labels" in batch else None
            else:
                batch_idx = batch["idx"][self.local_rank*self.batch_size_per_device:(self.local_rank+1)*self.batch_size_per_device]
                input_ids = batch["input_ids"][self.local_rank*self.batch_size_per_device:(self.local_rank+1)*self.batch_size_per_device]
                attention_mask = batch["attention_mask"][self.local_rank*self.batch_size_per_device:(self.local_rank+1)*self.batch_size_per_device]
                labels = batch["labels"][self.local_rank*self.batch_size_per_device:(self.local_rank+1)*self.batch_size_per_device] if "labels" in batch else None

            batch_idx = batch_idx.to(f"cuda:{self.local_rank}")
            input_ids = input_ids.to(f"cuda:{self.local_rank}")
            attention_mask = attention_mask.to(f"cuda:{self.local_rank}")
            labels = labels.to(f"cuda:{self.local_rank}") if labels is not None else None

            with torch.no_grad():
                outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
                results = self._probe(outputs, attention_mask, labels = labels)
            
            sample_idx = batch_idx.tolist()
            sample_dict = [{self.field: lst_hs, "idx": s_id} for lst_hs, s_id in zip(results.tolist(), sample_idx)]
            
            if(self.world_size > 1):
                all_process_embeddings = [[] for _ in range(self.world_size)]
                dist.gather_object(sample_dict, all_process_embeddings if dist.get_rank() == 0 else None, dst=0)
            else:
                all_process_embeddings = [sample_dict]

            if self.local_rank == 0:
                if b_idx == total_batches - 1:
                    for process_list in all_process_embeddings[:last_batch_size]:
                        all_embeddings_list.extend(process_list)
                else:
                    for process_list in all_process_embeddings:
                        all_embeddings_list.extend(process_list)     
            
            if b_idx == total_batches - 1:
                dist.barrier()
                
            del outputs
            del results
        
        return all_embeddings_list