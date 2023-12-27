from deita.selection.scorer import Llama_Scorer

model_name_or_path = "hkust-nlp/deita-quality-scorer"

scorer = Llama_Scorer(model_name_or_path)

# example input
input_text = "word to describe UI with helpful tooltips" # Example Input
output_text = "User-friendly or intuitive UI" # Example Output
quality_score = scorer.infer_quality(input_text, output_text)

print(quality_score)