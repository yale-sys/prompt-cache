from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

models = [
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "mosaicml/mpt-7b-chat-8k",
    "WizardLM/WizardLM-7B-V1.0",
    "WizardLM/WizardLM-13B-V1.0",
    "lmsys/vicuna-13b-v1.5-16k",
    "lmsys/vicuna-7b-v1.5-16k",
    "lmsys/longchat-7b-v1.5-32k",
    "internlm/internlm-chat-7b",

]

for model in models:
    print(model)
    model_path = model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
