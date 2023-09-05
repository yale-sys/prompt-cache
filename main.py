from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,

)

from promptcache import Schema, Prompt, CompactSpaces, FormatLlama2Conversation, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters

model_path = "meta-llama/Llama-2-7b-chat-hf"


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

    schema_text = read_file("./benchmark/schema_mbti_short.xml", preproc)
    prompt_text = read_file("./benchmark/prompt_mbti.xml", preproc)

    schema = Schema(schema_text, tokenizer)
    prompt = Prompt(prompt_text)

    print(schema)
    print(prompt)

    cache_engine.add_schema(schema)

    parameter = GenerationParameters(
        temperature=1.0,
        repetition_penalty=1.17,
        top_p=0.95,
        top_k=50,
        max_new_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id],
    )

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

        prompt = Prompt(prompt_text)
        token_ids, position_ids, cache = cache_engine.process(prompt)

        output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2)

        print(f"Assistant: ", end="", flush=True)

        resp = ""

        for outputs in output_stream:
            output_text = outputs.new_text.strip().split(" ")
            resp += outputs.new_text
            print(" ".join(output_text), end=" ", flush=True)

        prompt_text += f"<assistant>{resp}</assistant>"


if __name__ == "__main__":
    main()
