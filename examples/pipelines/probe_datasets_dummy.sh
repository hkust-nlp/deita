export HF_HOME="/data/data7/models"

GPUIDX="0,1,2,3"
# NUMPROCESS=1
# DATAPATH="/home/LeiFeng/weiliu/openai-toolkits/data/instag_mix_clean_multi_turn_infer_llama_reward_combine_all_npy_score_clean.json"
DATAPATH="data/deita_mix_dummy_101.json"
BSZ=1
OUTPUTPATH="outputs/mixtral_loss.pickle"
MODELPATH="mistralai/mixtral-8X7B-v0.1"
MASTER_ADDR="localhost"
MASTER_PORT=12355

CUDA_VISBLE_DEVICES=$GPUIDX accelerate launch \
    --config_file src/deita/ds_configs/accelerate_fsdp.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --fsdp_transformer_layer_cls_to_wrap MixtralSparseMoeBlock \
    --num_processes 4 \
    examples/pipelines/probe_datasets_fsdp.py \
    --use_flash_attention True \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --batch_size_per_device $BSZ \
    --mask_user true \
    --model_name_or_path $MODELPATH \
    --max_length 2048
