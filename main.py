import torch.cuda

from promptcache.model import Llama2, Falcon, Mpt
from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,

)

from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template


def main():
    ### Configurations ###

    disable_cuda = False
    disable_prompt_cache = False

    ######################

    # lm = Llama2("meta-llama/Llama-2-13b-chat-hf",
    #             load_in_8bit=True if not disable_cuda else False,
    #             device_map="auto" if not disable_cuda else None)

    # lm = Falcon("tiiuae/falcon-7b-instruct",
    #             load_in_8bit=True if not disable_cuda else False,
    #             device_map="auto" if not disable_cuda else None)

    lm = Mpt("mosaicml/mpt-7b-chat-8k",
             load_in_8bit=True if not disable_cuda else False,
             device_map="auto" if not disable_cuda else None)

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
    # 14649
    # 16869
    # torch.cuda.synchronize()
    # print(f'Mem: {torch.cuda.memory_allocated(0) / (1e6):.2f} MB')

    # cache_engine.add_schema(read_file("./benchmark/schema_mbti.xml", preproc))
    cache_engine.add_schema(read_file("./benchmark/schema_persona_long.xml", preproc))
    cache_engine.add_schema(read_file("./benchmark/empty.xml", preproc))

    # torch.cuda.synchronize()
    # print(f'Mem: {torch.cuda.memory_allocated(0) / (1e6):.2f} MB')

    parameter = GenerationParameters(
        temperature=0.1,
        repetition_penalty=1.17,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    # prompt_text = "<prompt schema='mbti'> <E/><N/><T/><P/>"
    prompt_text = """
    <prompt schema='persona'>
        <age>
            <young-adult/>
        </age>
        <residence>
            <seaside/>
        </residence>
        <education>
            <doctorate/>
        </education>
        <occupation>
            <technology/>
        </occupation>
        <martial-status>
            <married/>
        </martial-status>
        <personality>
            <introverted/>
        </personality>
    """

    # prompt_text = """
    #     <prompt schema='empty'>
    #     """

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
        token_ids, position_ids, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache)
        if disable_prompt_cache:
            assert cache is None

        output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2)

        txt = lm.decode(token_ids)

        # print(txt)

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
    main()
