
SCORETYPE="complexity"
DATAPATH="data/deita_mix_100.json"
OUTPUTPATH="outputs/deita_mix_complexity/deita_mix_complexity_mistral_sampled.json"
MODELPATH="/data/data9/outputs/complexity-scorer-mistral-z"
SCORER="mistral"
ISVLLM=false

python examples/scoring/score_complexity_dataset.py \
    --data_path $DATAPATH \
    --output_path $OUTPUTPATH \
    --score_type $SCORETYPE \
    --scorer $SCORER \
    --scorer_name_or_path $MODELPATH