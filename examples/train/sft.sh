export WANDB_PROJECT="Deita"
RUNNAME="Deita-7B-SFT"
MODELPATH="mistralai/Mistral-7B-v0.1"
DATAPATH="hkust-nlp/deita-6k-v0"
MODEL_SIZE="7B"
OUTPUTPATH="/PATH/TO/OUTPUTS"
DEVICES=""  # e.g. 0,1,2,3
NUMGPUS=$(echo $DEVICES | awk -F',' '{print NF}')
TOTALBSZ=512
BSZPERDEV=1
GPUIDX=0
GRADACC=$(($TOTALBSZ/$NUMGPUS/$BSZPERDEV))
echo "Training mistral model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BSZPERDEV batch size per GPU, $GRADACC gradient accumulation steps"

deepspeed --include localhost:${DEVICES} --master_port 29501 src/deita/alignment/train.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 6 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --evaluation_strategy "no" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --conv_template "vicuna_v1.1" \
    --mask_user True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --deepspeed src/deita/ds_configs/deepspeed_config_zero2_no_offload.json