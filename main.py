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

    use_cache = False

    cache_engine.add_schema(read_file("./benchmark/squadv2test.xml", preproc))

    parameter = GenerationParameters(
        temperature=0.0,
        repetition_penalty=1.17,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompt_text = """
    <prompt schema='squadv2'>
        <contexts>
            <document0/>
            <document1/>
            <document2/>
            <document3/>
            <document4/>
            <document5/>
            <document6/>
        </contexts>
    """

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

        print(f"Assistant: ")

        resp = ""
        final_output = None
        pre = 0
        for outputs in output_stream:
            final_output = outputs
            output_text = outputs.new_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                tt = " ".join(output_text[pre:now])
                resp += tt + " "
                print(tt, end=" ", flush=True)
                pre = now
        #print(resp)
        print("\n")
        prompt_text += f"<assistant>{resp}</assistant>"


if __name__ == "__main__":
    main()
