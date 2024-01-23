import os
import torch
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    )
from deita.selection.embedder.utils import batchlize
from transformers.trainer_pt_utils import LabelSmoother
from accelerate import Accelerator

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        
        
class Embedder:
    
    def __init__(self, model_name_or_path, **kwargs) -> None:
        
        self.compute_dtype = (
        torch.float16 
        if kwargs.get('fp16', False)
        else (torch.bfloat16 if kwargs.get("bfloat16", False) else torch.float32)
        )

        self.model_name_or_path = model_name_or_path
        self.max_length = kwargs.get('max_length')
        self.use_flash_attention = kwargs.get('use_flash_attention')
        self.batch_size_per_device = kwargs.get('batch_size_per_device')
        self.conv_template = kwargs.get('conv_template')
        self.only_answer = kwargs.get('only_answer')
        self.random_shuffle = kwargs.get('random_shuffle')
        
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
        
        self.accelerator = Accelerator()
        self.accelerator.wait_for_everyone()
        
        batch_size = self.batch_size_per_device  * self.world_size
        self.minibatch_size = batch_size
                
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, 
                                                        model_max_length = self.max_length,
                                                        padding_side = "right",
                                                        use_fast = False)
        
        if "mistral" in self.model_name_or_path:
            self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path,
                                               torch_dtype = self.compute_dtype,
                                               use_flash_attention_2 = self.use_flash_attention)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

    def rank0_print(self, *args):
        if self.local_rank == 0:
            print(*args)
            
    def compute_length(self, conversations: list, cnt_field = "response"):
        
        all_lengths = []
        for conv in conversations:

            cur_length = 0            
            
            for i, c in enumerate(conv):
                if cnt_field == "response":
                    if i % 2 == 1:
                        cur_length += len(c["value"])
                elif cnt_field == "instruction":
                    if i % 2 == 0:
                        cur_length += len(c["value"])
                else:
                    cur_length += len(c["value"])
                            
            all_lengths.append(cur_length)
        
        return all_lengths
    
    def create_databuffer(self, conversations: list, sort_by_length = False):
        
        all_lengths = self.compute_length(conversations)
        
        dataset_size = len(conversations)
        dataset_buf = []
        for idx in range(dataset_size):

            dataset_buf.append({
                "conversations": conversations[idx],
                "specific_length": all_lengths[idx],
                "input_idx": idx
            })
        
        if sort_by_length:
            dataset_buf = sorted(dataset_buf, key = lambda x: x["specific_length"])            
            
        return dataset_buf, dataset_size
    
    def create_dataloader(self, dataset_buf: Dataset):
        
        dataloader = batchlize(
            dataset_buf,
            self.minibatch_size,
            self.random_shuffle,
            sort = self.minibatch_size > 1
        )
        
        print(f"Successfully create dataloader with size {len(dataloader)},batch_size {self.minibatch_size}.")
        
        return dataloader
    
    def probe_samples(self, model, data: list):
        
        raise NotImplementedError

    def collect_grad(self):

        raise NotImplementedError