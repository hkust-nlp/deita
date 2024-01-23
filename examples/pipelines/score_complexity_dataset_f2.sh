
FOLD=2
SCORETYPE="complexity"
DATAPATH="data/split/instag_mix_clean_multi_turn_infer_llama_reward_combine_all_npy_score_clean_f${FOLD}.json"
OUTPUTPATH="outputs/deita_mix_complexity/deita_mix_mistral_f${FOLD}.json"
MODELPATH="/data/data9/outputs/complexity-scorer-mistral-z"
SCORER="mistral"
ISVLLM=true

CUDA_VISIBLE_DEVICES=$FOLD python examples/scoring/score_complexity_dataset.py \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --score_type $SCORETYPE \
    --scorer $SCORER \
    --scorer_name_or_path $MODELPATH \
    --is_vllm $ISVLLM