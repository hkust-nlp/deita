
SCORETYPE="quality"
DATAPATH="" # PATH/TO/DATASET
OUTPUTPATH=""  # PATH/TO/OUTPUTS
MODELPATH=""    # PATH/TO/MODEL
SCORER="mistral"
ISVLLM=true
GPUINDICES="0,1,2,3"

CUDA_VISIBLE_DEVICES=$GPUINDICES python examples/pipelines/score_complexity_dataset.py \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --score_type $SCORETYPE \
    --scorer $SCORER \
    --scorer_name_or_path $MODELPATH \
    --is_vllm $ISVLLM