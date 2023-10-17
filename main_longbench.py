import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse


# This is the customized building prompt for chat models
def build_chat(prompt):
    prompt = f"[INST]{prompt}[/INST]"

    return prompt


def main():
    path = 'meta-llama/Llama-2-7b-chat-hf'
    max_length = 3500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LlamaForCausalLM.from_pretrained(path, load_in_8bit=True, device_map="auto")
    tokenizer = LlamaTokenizer.from_pretrained(path)

    dataset = load_dataset('THUDM/LongBench', "narrativeqa", split='test')

    prompt_format = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    max_gen = 3500

    sample = dataset[0]

    prompt = prompt_format.format(**sample)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized_prompt[-half:], skip_special_tokens=True)

    prompt = build_chat(prompt)

    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

    print(prompt)

    context_length = input.input_ids.shape[-1]

    # print(input.input_ids)

    torch.save(list(input.input_ids.cpu().numpy()), 'input_ids.pt')

    print('context length', context_length)

    output = model.generate(
        **input,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        top_p=None,
        temperature=1.0,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    print(pred)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)

    main()
