import torch.cuda
import fire
import sys, json
import os
import datetime
# eval_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.abspath(os.path.join(eval_path, '..')))
from promptcache.model import Llama2, Falcon, Mpt
from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
)
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template

from benchmark.benchmark_base import Benchmark, Entry, DATASET_LIST, SCHEMA_FILE_DIRECTORY
from benchmark.squad_v2 import SquadV2
from benchmark.multi_news import MultiNews
from benchmark.ms_marco_v1_1 import MSMarcoV1

BENCHMARK_PATH = "./benchmark"

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
        
        self.cache_engine = CacheEngine(self.llm_config.get("max_ctx_length", 4096), self.lm)
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

        if dataset is None or dataset not in DATASET_LIST:
            raise ValueError("Dataset name cannot be None, valid dataset names are: " + ", ".join(DATASET_LIST))
        elif "squad" in dataset:
            self.dataset = SquadV2()
        elif "multi_news" in dataset:
            self.dataset = MultiNews()
        elif "wiki" in dataset:
            pass
        elif "pubmed" in dataset:
            pass
        elif "ms_marco" in dataset:
            self.dataset = MSMarcoV1()
        
        # for testing purpose, limit the entries to a small number
        self.dataset.init(limit_entries=3)

        # create result directory
        self.result_directory = os.path.join(BENCHMARK_PATH, "results", f"{self.model_name}-{self.dataset.dataset_name}", datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
        if not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory)

    def store_results(self, results):
        if self.enable_cache:
            prefix = "with_cache"
        else:
            prefix = "no_cache"
        with open(os.path.join(self.result_directory, f"{prefix}_results.json"), "a") as f:
            json.dump(results, f)
            f.write("\n")

    def run(self, batch_cache_size=1):
        entry_count = self.dataset.get_entry_count()
        for i in range(0, entry_count, batch_cache_size):
            entries = self.dataset.get_query((i, i + batch_cache_size))
            # load schema for `batch_cache_size` entries
            for entry in entries:
                schema_file_path = os.path.join(SCHEMA_FILE_DIRECTORY, self.dataset.dataset_name, entry.schema)
                print(schema_file_path)
                self.cache_engine.add_schema(read_file(schema_file_path, self.preproc), batch_size=self.llm_config.get("schema_load_batch", 1))

            for entry in entries:
                prompt = Prompt(entry.prompt, self.preproc)
                print(entry.prompt)
                no_cache = not self.enable_cache
                token_ids, position_ids, cache_time, cache = self.cache_engine.process(prompt, no_cache=no_cache,
                                                              return_full_position_ids=self.lm.use_full_position_ids)
                if no_cache:
                    assert cache is None

                output_stream = self.gen_engine.generate(token_ids, position_ids, self.parameter, cache, stream_interval=2,
                                            use_full_position_ids=self.lm.use_full_position_ids)
            
                resp = ""
                pre = 0
                response_time = 0.0
                for outputs in output_stream:
                    response_time = outputs.response_time
                    output_text = outputs.new_text.strip().split(" ")
                    now = len(output_text) - 1
                    if now > pre:
                        tt = " ".join(output_text[pre:now])
                        resp += tt + " "
                        print(tt, end=" ", flush=True)
                        pre = now

                result = {
                    "cache_time": cache_time,
                    "response_time": response_time,
                    "answer": entry.answer,
                    "response": resp
                }
                self.store_results(result)
                print("\n")

            self.cache_engine.remove_all_schemas()

def main(llm_config_path: str="config/llm_config_llama2.json", dataset: str="ms_marco", enable_cache=False):
    eval = Eval(llm_config_path, dataset, enable_cache)
    eval.run()

if __name__ == "__main__":
    fire.Fire(main)