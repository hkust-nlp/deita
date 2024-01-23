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
                    data_path = args.data_path, 
                    scorer = args.scorer, 
                    scorer_name_or_path = args.scorer_name_or_path, 
                    is_vllm = args.is_vllm, 
                    score_type = args.score_type, 
                    output_path = args.output_path)

pipeline.run()