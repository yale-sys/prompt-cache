from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,

)

from promptcache import Schema, Prompt, CompactSpaces, FormatLlama2Conversation, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template

model_path = "meta-llama/Llama-2-13b-chat-hf"


def main():
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path,
                                             load_in_8bit=True,
                                             device_map="auto")
    cache_engine = CacheEngine(model, tokenizer)
    gen_engine = GenerationEngine(model, tokenizer)

    preproc = [
        CompactSpaces(),
        FormatLlama2Conversation()
    ]

    cache_engine.add_schema(read_file("./benchmark/schema_mbti.xml", preproc))
    cache_engine.add_schema(read_file("./benchmark/schema_persona_long.xml", preproc))

    parameter = GenerationParameters(
        temperature=0.1,
        repetition_penalty=1.17,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompt_text = "<prompt schema='mbti'> <E/><N/><T/><P/>"
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

    use_cache = False

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

        token_ids, position_ids, cache, orig_token_ids, orig_pos_ids = cache_engine.process(prompt)

        if use_cache:
            output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2)

        else:
            output_stream = gen_engine.generate(orig_token_ids, orig_pos_ids, parameter, cache=None, stream_interval=2)

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
