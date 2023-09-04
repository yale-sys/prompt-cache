from transformers import (
    AutoTokenizer,

)

from promptcache import Schema, Prompt, CompactSpaces, FormatLlama2Conversation, read_file

model_path = "meta-llama/Llama-2-13b-chat-hf"


# cached conversation template


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    preproc = [
        CompactSpaces(),
        FormatLlama2Conversation()
    ]

    schema_raw = read_file("./benchmark/schema_mbti_short.xml", preproc)
    prompt_raw = read_file("./benchmark/prompt_mbti.xml", preproc)

    print(tokenizer.tokenize("Sensing-Intuition. For the Sensing-Intuition indicator, "))

    schema = Schema(schema_raw, tokenizer)
    prompt = Prompt(prompt_raw)

    print(schema)
    print(prompt)


if __name__ == "__main__":
    main()
