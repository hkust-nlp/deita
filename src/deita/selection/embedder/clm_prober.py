import torch
from deita.selection.embedder import CLM_Embedder

import logging

logger = logging.getLogger(__name__)

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits_ = shift_logits.view(-1, vocab_size)
    shift_labels_ = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels_ = shift_labels_.to(shift_logits_.device)
    loss = loss_fct(shift_logits_, shift_labels_)
    
    loss = loss.view_as(shift_labels).mean(1)
    
    return loss

class CLM_Prober(CLM_Embedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.field = kwargs.get('field', "loss")
        self.mask_user = kwargs.get("mask_user", False)
        
    def _probe(self, outputs, attention_mask, **kwargs):
        
        labels = kwargs.get("labels")
        logits = outputs.logits
        
        loss = compute_loss(logits, labels, self.tokenizer.vocab_size)
        
        return loss