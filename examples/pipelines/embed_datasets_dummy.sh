GPUIDX="0,1,2,3"
NUMPROCESS=4
DATAPATH="data/deita_mix_dummy_101.json"
BSZ=1
OUTPUTPATH="outputs/dummy_embeds.pickle"

CUDA_VISIBLE_DEVICES=$GPUIDX accelerate launch \
    --mixed_precision bf16 \
    --num_processes $NUMPROCESS \
    --num_machines 1 \
    examples/pipelines/embed_datasets.py \
    --use_flash_attention true \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --batch_size_per_device $BSZ
