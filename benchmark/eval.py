import torch.cuda
import fire
import sys, json
import os
import datetime
sys.path.append('..')
from promptcache.model import Llama2, Falcon, Mpt
from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
)
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template

from benchmark_base import Benchmark

VALID_DATASET = ["squad_v2", "multi_news", "wiki_qa", "pubmed_qa", "ms_macro"]

class Eval():
    def __init__(self, llm_config_path, dataset, enable_cache):
        with open(llm_config_path, 'r') as f:
            self.llm_config = json.load(f)
        self.enable_cache = enable_cache

        self.model_name = self.llm_config["name"]
        if "llama" in self.model_name:
            self.model_name = "llama"
            self.lm = Llama2(**self.llm_config)
        elif "falcon" in self.model_name:
            self.model_name = "falcon"
            self.lm = Falcon(**self.llm_config)
        elif "mpt" in self.model_name:
            self.model_name = "mpt"
            self.lm = Mpt(**self.llm_config)
        else:
            raise ValueError("Invalid model name")
        
        self.cache_engine = CacheEngine(self.llm_config.get("max_ctx_length", 8192), self.lm)
        self.gen_engine = GenerationEngine(self.lm)
        self.preproc = [
            CompactSpaces(),
            self.lm.get_formatter()
        ]

        self.parameter = GenerationParameters(
            temperature=0.1,
            repetition_penalty=1.17,
            top_p=0.95,
            top_k=-1,
            max_new_tokens=512,
            stop_token_ids=self.lm.stop_token_ids,
            stop_str=self.lm.stop_str
        )

        if dataset is None:
            raise ValueError("Dataset name cannot be None, valid dataset names are: " + ", ".join(VALID_DATASET))
        elif "squad" in dataset:
            pass
        elif "news" in dataset:
            pass
        elif "wiki" in dataset:
            pass
        elif "pubmed" in dataset:
            pass
        elif "macro" in dataset:
            pass

    def run(self, dataset: Benchmark=None):
        
        for schema in dataset.get_documents():
            self.cache_engine.add_schema(read_file(schema, self.preproc), batch_size=self.dataset_config["schema_load_batch"])

        now = datetime.datetime.now()
        directory = os.path.join("./results", f"{self.model_name}-{self.dataset_name}", datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
        os.makedirs(directory)

def main(llm_config_path: str, dataset: str="squad_v2", enable_cache=True):
    eval = Eval(llm_config_path, dataset, enable_cache)
    # eval.run()

if __name__ == "__main__":
    fire.Fire(main)