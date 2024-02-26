import os
import json
import torch
import random
import numpy as np
from datasets import Dataset
from typing import Sequence, Dict, List
from dataclasses import dataclass
from deita.selection.embedder.conversation import get_conv_template, SeparatorStyle


IGNORE_INDEX=-100

def load_json(path: str) -> List[Dict]:
    """
    Load json file.
    
    Parameters
    ------------
    path : str
        Path to the json file.
    
    Returns
    ------------
    data : List[Dict]
        List of dict.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data
    
def preprocess(
        sample,
        conv_template,
        only_answer,
        max_length,
        tokenizer,
        **kwargs,
    ) -> Dict:

        mask_user = kwargs.get('mask_user', False)

        sample_ids = sample["input_idx"]    
        source = sample["conversations"]

        conv = get_conv_template(conv_template)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if not only_answer:
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                # assert role == conv.roles[j % 2], f"{i}"
                assert role == conv.roles[j % 2], breakpoint()
                conv.append_message(role, sentence["value"])
            conversation = conv.get_prompt()
        else:
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            messages = []
            for j, sentence in enumerate(source):
                if j % 2 == 0:
                    continue
                messages.append(sentence["value"])
            conversation = "\n".join(messages)

        input_ids = tokenizer(
            conversation,
            return_tensors="pt",
            max_length=max_length, 
            truncation=True,
        ).input_ids
        
        input_ids = input_ids.flatten()
        
        targets = None
        
        if mask_user:
            
            assert not only_answer, "Only support only_answer = False when mask_user = True"
        
            targets = input_ids.clone()

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                    
                role = roles[sentence["from"]]
                # assert role == conv.roles[j % 2], f"{i}"
                assert role == conv.roles[j % 2], breakpoint()
                
                if role == conv.roles[1]:
                    conv.append_message(role, sentence["value"])
                
                if role != conv.roles[1]:
                    if j == 0:
                        conv_start_idx = 0
                    else:
                        conv_last = conv.get_prompt()
                        conv_start_idx = tokenizer(
                            conv_last,
                            return_tensors="pt",
                            max_length=max_length, 
                            truncation=True,
                        ).input_ids.shape[1]
                        
                    conv.append_message(role, sentence["value"])
                    conv_so_far = conv.get_prompt()
                    
                    conv_end_idx = tokenizer(
                        conv_so_far,
                        return_tensors="pt",
                        max_length=max_length, 
                        truncation=True,
                    ).input_ids.shape[1]
                    
                    # conv_end_idx -= 1   # hard offset for llama model
                    
                    targets[conv_start_idx:conv_end_idx] = IGNORE_INDEX
                    
                    if conv_end_idx >= max_length:
                        break
        
        attention_mask = torch.ones_like(input_ids)
            
        if targets is not None:
            return dict(
                input_ids=input_ids,
                idx = sample_ids,
                labels = targets,
                attention_mask = attention_mask,
            )
        else:
            return dict(
                input_ids=input_ids,
                idx = sample_ids,
                attention_mask = attention_mask,
            )
            
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, = tuple([instance[key] for instance in instances] for key in ("input_ids",))
        input_ids = torch.tensor(input_ids)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        instance_index = torch.tensor([instance["idx"] for instance in instances]).to(input_ids.device)

        return dict(
            idx = instance_index,
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def set_random_seed(seed: int):
    """
    Set the random seed for `random`, `numpy`, `torch`, `torch.cuda`.

    Parameters
    ------------
    seed : int
        The default seed.
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def batchlize(examples: list, batch_size: int, random_shuffle: bool, sort: bool, length_field = 'specific_length'):
    """
    Convert examples to a dataloader.

    Parameters
    ------------
    examples : list.
        Data list.
    batch_size : int.

    random_shuffle : bool
        If true, the dataloader shuffle the training data.
    
    sort: bool
        If true, data will be sort by its input length
    Returns
    ------------
    dataloader:
        Dataloader with batch generator.
    """
    size = 0
    dataloader = []
    length = len(examples)
    if (random_shuffle):
        random.shuffle(examples)
    
    new_examples = examples
    if sort:
        new_examples = sorted(examples, key = lambda x: len(x[length_field]))
    
    while size < length:
        if length - size > batch_size:
            dataloader.append(new_examples[size : size+batch_size])
            size += batch_size
        else:
            dataloader.append(new_examples[size : size+(length-size)])
            size += (length - size)
    return dataloader

def get_emb_name(**kwargs):
    
    if kwargs.get('model_path'):
        model_path = kwargs.pop("model_path")
        return os.path.basename(model_path)
    
    if kwargs.get('emb_name'):
        emb_name = kwargs.pop('emb_name')
        return os.path.basename(emb_name)