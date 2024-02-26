import torch
import json
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import accelerate
import logging
INSTRUCTION_ONLY_TYPE = ["complexity"]
INSTRUCTION_RESPONSE_TYPE = ["quality"]

logger = logging.getLogger(__name__)
def sort_key_split(sort_key: str) -> List:
    """
    Split sort_key into a list of sort keys.
    
    sort_key: str - sort key to split.
    """
    if "," in sort_key:
        return sort_key.split(",")
    elif "." in sort_key:
        return sort_key.split(".")
    elif "+" in sort_key:
        return sort_key.split("+")
    else:
        raise ValueError("sort_key must be a string with delimiter ',' or '.' or '+'.")

def load_data(self, data_path: str) -> None:
    
    """
    Load data from data_path.
    
    data_path: str - path to json data file.
    """
    
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
    
    return data

def load_transformers_model(model_name_or_path, **kwargs):
    
    compute_dtype = kwargs.get("compute_dtype", torch.bfloat16)
    if isinstance(compute_dtype, str):
        compute_dtype = setattr(torch, compute_dtype)
    
    use_flash_attention = kwargs.get("use_flash_attention", False)
    
    if use_flash_attention:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = None
    
    if "mixtral" in model_name_or_path.lower():
        attn_implementation = None
        logger.warning("Mixtral in deita have not been supported to use flash attention 2 yet, closed")
        
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            torch_dtype = compute_dtype,
                                            attn_implementation=attn_implementation)

    return model

def load_tokenizer(model_name_or_path, **kwargs):
    
    max_length = kwargs.get("max_length", 2048)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                    model_max_length = max_length,
                                                    padding_side = "right",
                                                    use_fast = False)
    # if "mistral" in model_name_or_path.lower() or "mixtral" in model_name_or_path.lower():
    #     tokenizer.padding_side = "left"
    if "mistral" in model_name_or_path.lower():
        tokenizer.padding_side = "left"
    
    return tokenizer

def accelerator_and_fsdp_warp(model):
    
    import accelerate
    from peft.utils.other import fsdp_auto_wrap_policy
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    
    accelerator = accelerate.Accelerator()
    
    fsdp_plugin = accelerator.state.fsdp_plugin
    auto_wrap_policy = fsdp_auto_wrap_policy(model)
    kwargs = {
        "sharding_strategy": fsdp_plugin.sharding_strategy,
        "cpu_offload": fsdp_plugin.cpu_offload,
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": fsdp_plugin.mixed_precision_policy,
        "sync_module_states": fsdp_plugin.sync_module_states,
        "use_orig_params": False,  # this should be `False`
        "limit_all_gathers": True,
        "param_init_fn": fsdp_plugin.param_init_fn,
        "device_id": accelerator.device,
    }

    fsdp_model =  FSDP(model, **kwargs)
    fsdp_model.eval()
    
    return fsdp_model, accelerator
    
    