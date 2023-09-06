import gc
from dataclasses import dataclass
from typing import Optional, Generator, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from promptcache.cache_engine import KVCache

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


@dataclass
class GenerationParameters:
    temperature: float = 1.0
    repetition_penalty: float = 1.17
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 256
    stop_token_ids: Optional[List[int]] = None
    echo: bool = True

    def get_logits_processor(self):
        p = LogitsProcessorList()
        if self.temperature >= 1e-5 and self.temperature != 1.0:
            p.append(TemperatureLogitsWarper(self.temperature))
        if self.repetition_penalty > 1.0:
            p.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))
        if 1e-8 <= self.top_p < 1.0:
            p.append(TopPLogitsWarper(self.top_p))
        if self.top_k > 0:
            p.append(TopKLogitsWarper(self.top_k))
        return p


@dataclass
class Output:
    text: str
    new_text: str


class GenerationEngine:
    model: LlamaForCausalLM
    tokenizer: LlamaTokenizer

    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self,
                 token_ids: List[int],
                 position_ids: List[int],
                 params: GenerationParameters,
                 cache: Optional[KVCache] = None,
                 stream_interval: int = 2) -> Generator[Output, None, None]:

        logits_processor = params.get_logits_processor()
        output_ids = list(token_ids)
        new_output_ids = list()

        # create tensors
        token_ids = torch.tensor([token_ids], device=self.model.device, dtype=torch.long)
        position_ids = torch.tensor([position_ids], device=self.model.device, dtype=torch.long)

        if cache is not None:
            cache = [(cache[i][0].to(self.model.device), cache[i][1].to(self.model.device)) for i in
                     range(len(cache))]

        past_key_values = None
        for i in range(params.max_new_tokens):

            if past_key_values is None:

                out = self.model(input_ids=token_ids,
                                 position_ids=position_ids,
                                 past_key_values=cache,
                                 use_cache=True)

                logits = out.logits
                past_key_values = out.past_key_values
            else:

                new_token_ids = torch.tensor([[token]], device=self.model.device, dtype=torch.long)

                out = self.model(input_ids=new_token_ids,
                                 # position_ids=position_ids,
                                 past_key_values=past_key_values,
                                 use_cache=True)

                logits = out.logits
                past_key_values = out.past_key_values

            if params.repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor(
                    [output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            if params.temperature < 1e-5 or params.top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            new_output_ids.append(token)

            if token in params.stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == params.max_new_tokens - 1 or stopped:
                output = self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                new_output = self.tokenizer.decode(
                    new_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                new_output_ids = []
                yield Output(output, new_output)

            if stopped:
                break

        # clean
        # del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()
