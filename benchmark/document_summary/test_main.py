import torch.cuda
import fire

# Add the parent directory to the sys.path list
import os, sys
document_summary_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(document_summary_path, '../..')))

from promptcache.model import Llama2, Falcon, Mpt
from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
)
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template

def main(enable_cache=False):
    ### Configurations ###

    disable_cuda = False
    disable_prompt_cache = not enable_cache

    ######################

    lm = Llama2("meta-llama/Llama-2-7b-chat-hf",
                load_in_8bit=True if not disable_cuda else False,
                device_map="auto" if not disable_cuda else None)

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
    cache_engine = CacheEngine(2500, lm)
    gen_engine = GenerationEngine(lm)

    preproc = [
        CompactSpaces(),
        lm.get_formatter()
    ]

    cache_engine.add_schema(read_file("./benchmark/document_summary/schema_summary_sample.xml", preproc))

    parameter = GenerationParameters(
        temperature=0.1,
        repetition_penalty=1.17,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    prompt_text = "<prompt schema='document_summary'> <Document0/>"

    # text chat interface
    while True:
        try:
            inp = input("User: ")
        except EOFError:
            inp = ""

        if inp == "exit" or not inp:
            print("Terminating...")
            break

        prompt_text += f"<user>{inp}</user>"

        prompt = Prompt(prompt_text + "</prompt>", preproc)
        # print(prompt)
        token_ids, position_ids, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache,
                                                              return_full_position_ids=lm.use_full_position_ids)
        if disable_prompt_cache:
            assert cache is None

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

        print("\n")
        prompt_text += f"<assistant>{resp}</assistant>"

if __name__ == "__main__":
    fire.Fire(main)
