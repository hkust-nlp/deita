GPUIDX="0,1,2,3"
NUMGPUS=$(echo $GPUIDX | awk -F',' '{print NF}')
DATAPATH=""
OTHERDATA=""    # PATH/TO/EMBEDDING_FILE
OUTPUTPATH=""   # PATH/TO/OUTPUTS
THETA=0.9
DATASIZE=10
BSZ=1

CUDA_VISIBLE_DEVICES=$GPUIDX python examples/pipelines/combined_filter.py \
    --data_path $DATAPATH \
    --other_data_path $OTHERDATA \
    --output_path $OUTPUTPATH \
    --threshold $THETA \
    --data_size $DATASIZE \
    --is_compression true \
    --device 0
