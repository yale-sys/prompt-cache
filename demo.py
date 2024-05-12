import random
import re

import numpy as np
import torch.cuda
import fire

from promptcache.model import Llama2, Falcon, Mpt, CodeLlama

from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template


def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)


def main(enable_cache=True):
    enable_cpu_inference = False
    disable_prompt_cache = not enable_cache

    lm_for_cache = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                             load_in_8bit=True,
                             device_map="auto")

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                       load_in_8bit=False,
                       device_map=None)

    # lm = Falcon("tiiuae/falcon-7b-instruct",
    #             load_in_8bit=True if not disable_cuda else False,
    #             device_map="auto" if not disable_cuda else None)

    # lm = Mpt("mosaicml/mpt-7b-chat-8k",
    #          load_in_8bit=True if not disable_cuda else False,
    #          device_map="auto" if not disable_cuda else None)

    cache_engine = CacheEngine(5000, lm_for_cache, target_device='cpu' if enable_cpu_inference else None)
    gen_engine = GenerationEngine(lm)

    preproc = [
        # CompactSpaces(),
        lm.get_formatter()
    ]

    cache_engine.add_schema(read_file("./examples/code_generation_game.xml", preproc), max_tokens=800)

    parameter = GenerationParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    prompt_text = f"""
        <prompt schema='code-generation-game'>
        <unit.py/>
        <map.py/>
        <player.py/>
        <game.py/>
        <database.py/>
        <user>
            Create a main entry for the game:
        </user>
        </prompt>
        """

    prompt = Prompt(prompt_text, preproc)
    token_ids, position_ids, cache_time, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache,
                                                                      return_full_position_ids=lm.use_full_position_ids)

    output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2,
                                        use_full_position_ids=lm.use_full_position_ids)

    print(f"Assistant: ", end="", flush=True)

    resp = ""
    pre = 0
    for outputs in output_stream:
        output_text = outputs.new_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            tt = " ".join(output_text[pre:now])
            resp += tt + " "
            print(tt, end=" ", flush=True)
            pre = now
    tt = " ".join(output_text[pre:])
    print(tt, flush=True)
    resp += tt

    print("\n")
    prompt_text += f"<assistant>{resp}</assistant>"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    fire.Fire(main)
