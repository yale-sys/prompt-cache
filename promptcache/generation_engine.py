import gc
import time
from dataclasses import dataclass, field
from typing import Optional, Generator, List, Tuple

import torch

from promptcache.cache_engine import KVCache

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import termcolor
from promptcache.model import LanguageModel


@dataclass
class GenerationParameters:
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 256
    stop_token_ids: List[int] = field(default_factory=lambda: [])
    stop_str: List[str] = field(default_factory=lambda: [])
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


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


@dataclass
class Output:
    text: str
    new_text: str
    response_time: float = 0.0
    elapsed_time: float = 0.0


class GenerationEngine:
    lm: LanguageModel

    def __init__(self, lm: LanguageModel):
        self.lm = lm

    @torch.inference_mode()
    def generate(self,
                 token_ids: List[int],
                 position_ids: List[int],
                 params: GenerationParameters,
                 cache: Optional[KVCache] = None,
                 stream_interval: int = 2,
                 use_full_position_ids: bool = False) -> Generator[Output, None, None]:

        logits_processor = params.get_logits_processor()
        output_ids = list(token_ids)
        new_output_ids = list()

        device = self.lm.device

        position_offset = max(position_ids) + 1
        past_key_values = None
        new_token_id = 0

        inference_time = 0.0
        response_time = 0.0

        position_ids_og = position_ids

        for i in range(params.max_new_tokens):

            # initial phase
            if past_key_values is None:

                input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
                position_ids = torch.tensor([position_ids], device=device, dtype=torch.long)
                # print(len(position_ids[0]))

                # add redundant batch dim
                if cache is not None:
                    cache = [(k[0].unsqueeze(0), k[1].unsqueeze(0)) for k in cache]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                out = self.lm(input_ids=input_ids,
                              position_ids=position_ids,
                              past_key_values=cache,
                              use_cache=True)
                end.record()
                torch.cuda.synchronize()
                inference_time += start.elapsed_time(end)
                response_time = inference_time
                # print(f'Response time: {inference_time:.2f} ms')
                # pretty print using termcolor
                print(termcolor.colored(f'Prefill latency: {inference_time:.2f} ms', 'yellow'))

                logits = out.logits
                past_key_values = out.past_key_values

            else:
                # upload to the GPU
                input_ids = torch.tensor([[new_token_id]], device=device, dtype=torch.long)

                if use_full_position_ids:
                    position_ids = torch.tensor([position_ids_og + list(range(position_offset, position_offset + i))],
                                                device=device, dtype=torch.long)

                else:
                    position_ids = torch.tensor([[position_offset + i]], device=device, dtype=torch.long)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                out = self.lm(input_ids=input_ids,
                              position_ids=position_ids,
                              past_key_values=past_key_values,
                              use_cache=True)
                end.record()
                torch.cuda.synchronize()
                inference_time += start.elapsed_time(end)

                logits = out.logits
                past_key_values = out.past_key_values

            if params.repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=device)
            else:
                tmp_output_ids = None

            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]

            ccc = []

            if params.temperature < 1e-5 or params.top_p < 1e-8:  # greedy
                new_token_id = int(torch.argmax(last_token_logits))
                # _, indices = torch.topk(last_token_logits, 2)
                # ccc = [int(index) for index in indices.tolist()]

            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                new_token_id = int(torch.multinomial(probs, num_samples=1))
                # probs = torch.softmax(last_token_logits, dim=-1)
                # indices = torch.multinomial(probs, num_samples=2)
                # ccc = [int(token) for token in indices.tolist()]

            # new_token_id = ccc[1]

            output_ids.append(new_token_id)
            new_output_ids.append(new_token_id)

            # print(self.lm.decode([new_token_id]))

            if new_token_id in params.stop_token_ids:
                # print('Stopped', self.lm.decode([new_token_id]), self.lm.encode('.</s><s> '))
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == params.max_new_tokens - 1 or stopped:
                output = self.lm.decode(output_ids)
                new_output = self.lm.decode(new_output_ids)

                partially_stopped = False

                for each_stop in params.stop_str:
                    pos = new_output.rfind(each_stop, 0)
                    if pos != -1:
                        new_output = new_output[:pos]
                        stopped = True
                        break
                    else:
                        partially_stopped = is_partial_stop(output, each_stop)
                        if partially_stopped:
                            break

                if not partially_stopped:
                    yield Output(output, new_output, inference_time, response_time)

            if stopped:
                break

        # clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()
