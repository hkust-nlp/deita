import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class Scorer(object):
    
    def __init__(self, model_name_or_path: str, is_vllm: bool  = False, **kwargs):
        
        self.is_vllm = is_vllm
        
        if not is_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        else:
            
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(model_name_or_path)
            self.sampling_params = SamplingParams(max_tokens = 2, logprobs = 1000)
        
    def infer_score(self, user_input: str):

        max_length = 2
        
        if self.is_vllm:
            outputs = self.llm.generate(user_input, self.sampling_params)
            score_template = np.array([1,2,3,4,5,6])
            
            try:
                logprobs_list = outputs[0].outputs[0].logprobs[0]
            except IndexError:
                return 3.0
        else:
            input_ids = self.tokenizer.encode(user_input, return_tensors = "pt")
            outputs = self.model.generate(input_ids, max_new_tokens = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
            
            try:
                logprobs_list = outputs.scores[0][0]
            except IndexError:
                return 3.0
            
        score_logits = []
        score_template = np.array([1,2,3,4,5,6])
        for k in self.id2score:
            try:
                score_logits.append(logprobs_list[k])
            except KeyError:
                return 3.0
                
        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)
        score_npy = score_npy * score_template

        score_npy = np.sum(score_npy, axis=0)
        
        return score_npy
            
    def infer_complexity(self, input_text: str):
        
        complexity_template = self.complexity_template
        user_input = complexity_template.format(instruction=input_text)
        
        return self.infer_score(user_input)
        
    def infer_quality(self, input_text: str, resp_text: str):
        
        quality_template = self.quality_template
        user_input = quality_template.format(instruction=input_text, output=resp_text)
        
        return self.infer_score(user_input)

    @property
    def id2score(self):
        raise NotImplementedError
    
    @property
    def complexity_template(self):
        raise NotImplementedError
    
    @property
    def quality_template(self):
        raise NotImplementedError