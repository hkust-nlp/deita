from deita.selection.scorer import Llama_Scorer

model_name_or_path = "hkust-nlp/deita-complexity-scorer"

scorer = Llama_Scorer(model_name_or_path)

# example input
input_text = "write a performance review for a junior data scientist"
complexity_score = scorer.infer_complexity(input_text)

print(complexity_score)