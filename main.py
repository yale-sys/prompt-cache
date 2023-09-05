from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,

)

from promptcache import Schema, Prompt, CompactSpaces, FormatLlama2Conversation, read_file, CacheEngine

model_path = "meta-llama/Llama-2-7b-chat-hf"


# cached conversation template


def main():
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path,
                                             load_in_8bit=True,
                                             device_map="auto")
    engine = CacheEngine(model, tokenizer)

    preproc = [
        CompactSpaces(),
        FormatLlama2Conversation()
    ]

    schema_raw = read_file("./benchmark/schema_mbti_short.xml", preproc)
    prompt_raw = read_file("./benchmark/prompt_mbti.xml", preproc)

    schema = Schema(schema_raw, tokenizer)
    prompt = Prompt(prompt_raw)

    print(schema)
    print(prompt)

    engine.add_schema(schema)

    res = engine.process(prompt)


if __name__ == "__main__":
    main()
