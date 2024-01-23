GPUIDX="0,1,2,3"
NUMPROCESS=4
DATAPATH="data/deita_mix_100.json"
BSZ=1

export HF_HOME="/data/data7/models"

CUDA_VISIBLE_DEVICES=$GPUIDX accelerate launch \
    --mixed_precision bf16 \
    --num_processes $NUMPROCESS \
    --num_machines 1 \
    examples/pipelines/embed_datasets.py \
    --use_flash_attention true \
    --data_path $DATAPATH \
    --output_path "outputs/embedding/deita_mix_sample100.pickle" \
    --batch_size_per_device $BSZ
