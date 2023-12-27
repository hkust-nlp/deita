export WANDB_PROJECT="Deita"
RUNNAME="Deita-7B"
MODELPATH="/PATH/TO/SFT_MODEL"
MODEL_SIZE="7B"
DEVICES=""  # e.g. 0,1,2,3
NUMGPUS=$(echo $DEVICES | awk -F',' '{print NF}')

DPOEPOCH=9
JSONPATH="/PATH/TO/ultrafeedback_or_sampled_ultrafeedback"   # If you want to sample UltraFeedback dataset, please refer to our code src/deita/data/sample_ultrafeedback.py
OUTPUTPATH="/PATH/TO/OUTPUTS"
DATASPLIT="train"
TOTALBSZ=32
BSZPERDEV=1
GRADACC=$(($TOTALBSZ/$NUMGPUS/$BSZPERDEV))
echo "DPO Training mistral model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BSZPERDEV batch size per GPU, $GRADACC gradient accumulation steps"

deepspeed --include localhost:${DEVICES} --master_port 29502 src/deita/alignment/dpo_train.py \
    --model_name_or_path ${MODELPATH} \
    --json_path ${JSONPATH} \
    --data_split ${DATASPLIT} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs ${DPOEPOCH} \
    --beta 0.1 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --save_global_steps False \
    --eval_steps 50 \
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --do_eval False \
    --evaluation_strategy "no" \
    --model_max_length 2048 \
    --conv_template "vicuna_v1.1" \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed src/deita/ds_configs/stage3_no_offloading_accelerate.json