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

class Eval():
    def __init__(self, llm_config_path, dataset_config_path, enable_cache):
        with open(llm_config_path, 'r') as f:
            self.llm_config = json.load(f)
        with open(dataset_config_path, 'r') as f:
            self.dataset_config = json.load(f)
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

        self.dataset_name = self.dataset_config["name"]
        
        self.cache_engine = CacheEngine(self.dataset_config["max_ctx_length"], self.lm)
        self.gen_engine = GenerationEngine(self.lm)
        self.preproc = [
            CompactSpaces(),
            self.lm.get_formatter()
        ]

    def run(self, dataset: Benchmark=None):
        
        # for schema in dataset.get_documents():
        #     self.cache_engine.add_schema(read_file(schema, self.preproc), batch_size=self.dataset_config["schema_load_batch"])

        now = datetime.datetime.now()
        directory = os.path.join("./results", f"{self.model_name}-{self.dataset_name}", datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
        os.makedirs(directory)

    

def main(llm_config_path, dataset_config_path, enable_cache=True):
    eval = Eval(llm_config_path, dataset_config_path, enable_cache)
    eval.run()

if __name__ == "__main__":
    fire.Fire(main)