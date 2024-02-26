import logging
import argparse
import deepspeed
from deita.pipeline import Pipeline
from deita.pipeline.utils import load_tokenizer, load_transformers_model, accelerator_and_fsdp_warp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--batch_size_per_device", type=int, default=1)
parser.add_argument("--conv_template", type=str, default="vicuna_v1.1")
parser.add_argument("--use_flash_attention", type=bool, default=False)
parser.add_argument("--only_answer", type=bool, default=False)
parser.add_argument("--random_shuffle", type=bool, default=False)
parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--mask_user", type=bool, default=False)
parser.add_argument("--num_proc", type=int, default=1)
parser.add_argument("--ds_config_path", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=0)

args = parser.parse_args()


model = load_transformers_model(
                    model_name_or_path = args.model_name_or_path,
                    use_flash_attention = args.use_flash_attention)
tokenizer = load_tokenizer(
                    model_name_or_path = args.model_name_or_path,
                    max_length = args.max_length)

model, accelerator = accelerator_and_fsdp_warp(model)

embed_pipeline = Pipeline("probe_pipeline", 
                          model = model,
                          tokenizer = tokenizer,
                          data_path = args.data_path,   # json file with sharegpt format
                          output_path = args.output_path,  # output path (pickle format)
                          model_name_or_path = args.model_name_or_path,  # model name or path e.g. mistralai/Mistral-7B-v0.1
                          max_length = args.max_length,
                          use_flash_attention = args.use_flash_attention,  
                          batch_size_per_device = args.batch_size_per_device,
                          conv_template = args.conv_template,
                          only_answer = args.only_answer,
                          random_shuffle = args.random_shuffle,
                          bfloat16 = True,
                          mask_user = args.mask_user,
                          num_proc = args.num_proc,
                          ds_config_path = args.ds_config_path
                          )

embed_pipeline.run()
