from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from promptcache.cache_engine import KVCache


class InferenceEngine:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self,
                 token_ids: torch.Tensor,
                 position_ids: torch.Tensor,
                 cache: Optional[KVCache] = None)
