import logging
import argparse
from deita.pipeline import Pipeline

logger = logging.getLogger(__name__)
logger.info("Running score_pipeline")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--scorer", type=str, default="llama")
parser.add_argument("--scorer_name_or_path", type=str, default="hkust-nlp/deita-complexity-scorer")
parser.add_argument("--is_vllm", type=bool, default=False)
parser.add_argument("--score_type", type=str, default=None)
args = parser.parse_args()


pipeline = Pipeline("score_pipeline", 
                    data_path = args.data_path,   # json file with sharegpt format
                    scorer = args.scorer,   # [mistral, llama]
                    scorer_name_or_path = args.scorer_name_or_path,  # scorer name or path e.g. hkust-nlp/deita-complexity-scorer
                    is_vllm = args.is_vllm,  # launch with vllm [True, False]
                    score_type = args.score_type, # [complexity, quality]
                    output_path = args.output_path)  # output path (json format)

pipeline.run()