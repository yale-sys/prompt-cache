from transformers import (
    AutoTokenizer,

)

from promptcache import Schema, Prompt

model_path = "meta-llama/Llama-2-13b-chat-hf"


# cached conversation template

def read_file(filename) -> str:
    with open(filename, 'r') as f:
        return f.read()


def main():

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    layout = Schema(read_file("./benchmark/schema_mbti.xml"), tokenizer)

    prompt = Prompt(read_file("./benchmark/prompt_mbti.xml"))

    print(layout.name)
    print(layout)

    print(prompt)


if __name__ == "__main__":
    main()
