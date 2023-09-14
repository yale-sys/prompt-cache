from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

models = [
    "BAAI/AquilaChat-7B",
    "baichuan-inc/Baichuan2-7B-Chat",
    "bigscience/bloomz",
    "tiiuae/falcon-7b",
    "gpt2",
    "bigcode/starcoder",
    "EleutherAI/gpt-j-6b",
    "databricks/dolly-v2-12b",
    "internlm/internlm-chat-7b",
    "mosaicml/mpt-7b",
    "Qwen/Qwen-7B-Chat"

]

for model in models:
    print(model)
    model_path = model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        print("OK")
    except:
        print("FAILED")
