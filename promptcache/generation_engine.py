import gc
import time
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
    elapsed_time: float = 0.0


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

        device = self.model.device

        position_offset = max(position_ids) + 1

        # upload cache to GPU
        if cache is not None:
            cache = [(cache[i][0].to(device), cache[i][1].to(device)) for i in range(len(cache))]

        past_key_values = None
        new_token_id = 0

        inference_time = 0.0

        for i in range(params.max_new_tokens):

            # initial phase
            if past_key_values is None:

                # upload to the GPU

                input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
                position_ids = torch.tensor([position_ids], device=device, dtype=torch.long)
                use_cache = True

                ffff = None
                if cache is not None:
                    ffff = [(cache[i][0].to(device), cache[i][1].to(device)) for i in range(len(cache))]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()

                start.record()
                out = self.model(input_ids=input_ids,
                                 position_ids=position_ids,
                                 past_key_values=ffff,
                                 use_cache=use_cache)
                end.record()
                torch.cuda.synchronize()
                inference_time += start.elapsed_time(end)

                print('Initial response time: ', inference_time)

                logits = out.logits
                past_key_values = out.past_key_values

                del ffff
            else:

                # upload to the GPU
                input_ids = torch.tensor([[new_token_id]], device=device, dtype=torch.long)
                position_ids = torch.tensor([[position_offset + i]], device=device, dtype=torch.long)
                t1 = time.time()
                out = self.model(input_ids=input_ids,
                                 position_ids=position_ids,
                                 past_key_values=past_key_values,
                                 use_cache=True)
                inference_time += time.time() - t1

                logits = out.logits
                past_key_values = out.past_key_values

            if params.repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=device)
            else:
                tmp_output_ids = None

            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            if params.temperature < 1e-5 or params.top_p < 1e-8:  # greedy
                new_token_id = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                new_token_id = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(new_token_id)
            new_output_ids.append(new_token_id)

            if new_token_id in params.stop_token_ids:
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
                yield Output(output, new_output, inference_time)

            if stopped:
                break

        # clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()
