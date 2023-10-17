import random
import re
import sys

import numpy as np
import torch.cuda
import fire
from datasets import load_dataset

from promptcache.model import Llama2, Falcon, Mpt, CodeLlama
from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
)
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template
from promptcache.prompt import apply_preproc


def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    # The lambda function ensures only the first letter is capitalized
    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)
    # return input_str.replace('<', '(').replace('>', ')')


def main(enable_cache=True):
    ### Configurations ###

    disable_cuda = False

    enable_cpu_inference = False

    disable_prompt_cache = not enable_cache

    ######################

    # lm_for_cache = Llama2("meta-llama/Llama-2-13b-chat-hf",
    #                       load_in_8bit=True,
    #                       device_map="auto")

    lm_for_cache = CodeLlama("codellama/CodeLlama-13b-Instruct-hf",
                             load_in_8bit=True,
                             device_map="auto")

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = Llama2("meta-llama/Llama-2-13b-chat-hf",
                    load_in_8bit=False,
                    device_map=None)

    # lm = Falcon("tiiuae/falcon-7b-instruct",
    #             load_in_8bit=True if not disable_cuda else False,
    #             device_map="auto" if not disable_cuda else None)

    # lm = Mpt("mosaicml/mpt-7b-chat-8k",
    #          load_in_8bit=True if not disable_cuda else False,
    #          device_map="auto" if not disable_cuda else None)

    # tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # model = LlamaForCausalLM.from_pretrained(model_path,
    #                                          load_in_8bit=True if not disable_cuda else False,
    #                                          device_map="auto" if not disable_cuda else None)

    # dataset = load_dataset('THUDM/LongBench', 'narrativeqa', split='test')
    #
    # sample = dataset[5]
    #
    # sample_context = escape_tags(sample["context"])
    # sample_input = sample["input"]
    #
    #     schema = f"""
    # <schema name="qa"><module name="context"><system/><user>You are given a story, which can be either a novel or a movie script, and a question. Answer the question as
    # concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory:{sample_context}</module></schema>
    #     """

    cache_engine = CacheEngine(7000, lm_for_cache)
    gen_engine = GenerationEngine(lm)

    preproc = [
        # CompactSpaces(),
        lm.get_formatter()
    ]
    # 14649
    # 16869
    # torch.cuda.synchronize()
    # print(f'Mem: {torch.cuda.memory_allocated(0) / (1e6):.2f} MB')

    cache_engine.add_schema(read_file("./examples/code_generation_game.xml", preproc))
    #cache_engine.add_schema(read_file("./examples/personalization-education.xml", preproc))

    # cache_engine.add_schema(apply_preproc(schema, preproc), max_tokens=3500)

    # torch.cuda.synchronize()
    # print(f'Mem: {torch.cuda.memory_allocated(0) / (1e6):.2f} MB')

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
    #
    # prompt_text = f"""
    #     <prompt schema='personalization-education'>
    #     <middle-school/>
    #     <beginner/>
    #     <studied-a-year-before/>
    #     <auditory/>
    #     <essay/>
    #     <high-intrinsic-motivation/>
    #     <user>
    #         Concisely describe the learner:
    #     </user>
    #     </prompt>
    #     """

    prompt = Prompt(prompt_text, preproc)
    ##print(prompt)
    token_ids, position_ids, cache_time, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache,
                                                                      return_full_position_ids=lm.use_full_position_ids)

    # # token_ids2 = torch.load('input_ids.pt')[0]
    #
    # # print(np.array(token_ids[-100:]))
    # # print(lm.decode(token_ids))
    # # print('---------------' * 4)
    # # print(token_ids2[-100:])
    # # print(lm.decode(token_ids2))
    #
    # ctx_len = len(token_ids)
    #
    # print(lm.decode(token_ids))
    #
    # if disable_prompt_cache:
    #     assert cache is None
    #
    # output = lm.hf_model.generate(
    #     inputs=torch.tensor([token_ids], device=lm.device, dtype=torch.long),
    #     max_new_tokens=256,
    #     num_beams=1,
    #     do_sample=False,
    #     top_p=None,
    #     temperature=1.0,
    # )[0]
    #
    # pred = lm.decode(output[ctx_len:])
    # print(pred)

    # position_ids = position_ids[:len(token_ids)]
    # position_ids = list(range(len(token_ids)))
    # print(position_ids)

    # print(position_ids[:len(token_ids)] == list(range(len(token_ids))))

    # position_ids = list(range(len(token_ids)))

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
