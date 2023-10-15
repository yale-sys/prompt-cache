import torch.cuda
import fire
import sys, json
import os
import datetime
from tqdm import tqdm
from benchmark.longbench import LongBench
from promptcache.model import Llama2, Falcon, Mpt
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters

from benchmark.benchmark_base import SCHEMA_FILE_DIRECTORY

BENCHMARK_PATH = "./benchmark"


class Eval:
    def __init__(self, llm_config_path, enable_cache, use_cpu_for_inference=False):
        with open("./config/dataset_maxlen.json", 'r') as f:
            self.dataset_maxlen = json.load(f)

        with open(llm_config_path, 'r') as f:
            self.llm_config = json.load(f)
        self.enable_cache = enable_cache
        self.use_cpu_for_inference = use_cpu_for_inference

        self.model_name = self.llm_config["name"]
        if "llama" in self.model_name:
            self.model_name = "llama"
            self.lm_for_caching = Llama2(**self.llm_config)
        elif "falcon" in self.model_name:
            self.model_name = "falcon"
            self.lm_for_caching = Falcon(**self.llm_config)
        elif "mpt" in self.model_name:
            self.model_name = "mpt"
            self.lm_for_caching = Mpt(**self.llm_config)
        else:
            raise ValueError("Invalid model name")

        if self.use_cpu_for_inference:
            if "llama" in self.model_name:
                self.lm = Llama2(name=self.llm_config['name'], device_map=None)
            elif "falcon" in self.model_name:
                self.lm = Falcon(name=self.llm_config['name'], device_map=None)
            elif "mpt" in self.model_name:
                self.lm = Mpt(name=self.llm_config['name'], device_map=None)
        else:
            self.lm = self.lm_for_caching

        self.cache_engine = CacheEngine(self.llm_config.get("max_ctx_length", 5000), self.lm_for_caching,
                                        target_device=self.lm.device)
        self.gen_engine = GenerationEngine(self.lm)
        self.preproc = [
            CompactSpaces(),
            self.lm.get_formatter()
        ]

        self.dataset_list = {
            "narrativeqa": LongBench("narrativeqa"),
            "qasper": LongBench("qasper"),
            "multifieldqa_en": LongBench("multifieldqa_en"),
            "hotpotqa": LongBench("hotpotqa"),
            "2wikimqa": LongBench("2wikimqa"),
            "musique": LongBench("musique"),
            "dureader": LongBench("dureader"),
            "gov_report": LongBench("gov_report"),
            "qmsum": LongBench("qmsum"),
            "multi_news": LongBench("multi_news"),
            "vcsum": LongBench("vcsum"),
            "trec": LongBench("trec"),
            "triviaqa": LongBench("triviaqa"),
            "samsum": LongBench("samsum"),
            "passage_count": LongBench("passage_count"),
            "passage_retrieval_en": LongBench("passage_retrieval_en"),
            "lcc": LongBench("lcc"),
            "repobench-p": LongBench("repobench-p")
        }

    # recomputation overhead vs mem trasnfer overhead
    # @torch.inference_mode()
    # def run_critical_point(self):
    #
    #     NUM_LAYERS = 30
    #     CACHE_DIM = (40, SEQ_LEN, 128)
    #
    #     def create_cache(device, seq_len):
    #
    #         return [(torch.rand((40, seq_len, 128), dtype=torch.float16, device=device),
    #                  torch.rand((40, seq_len, 128), dtype=torch.float16, device=device)) for _ in
    #                 range(30)]
    #
    #     for seq_len in range(5000):
    #
    #     def benchmark_transfer(src_cache, dst_cache, description):
    #         start_time = time.time()
    #         for src, dst in zip(src_cache, dst_cache):
    #             dst[0].copy_(src[0], non_blocking=True)
    #             dst[1].copy_(src[0], non_blocking=True)
    #         torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
    #         elapsed = (time.time() - start_time) / NUM_LAYERS
    #         print(f"{description} Average Latency: {elapsed * 1000:.2f} milliseconds")
    #
    #     input_ids = torch.tensor([token_ids], device=self.lm.device, dtype=torch.long)
    #     position_ids = torch.tensor([position_ids], device=self.lm.device, dtype=torch.long)
    #     # print(len(position_ids[0]))
    #
    #     # add redundant batch dim
    #     if cache is not None:
    #         cache = [(k[0].unsqueeze(0), k[1].unsqueeze(0)) for k in cache]
    #
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #
    #     start.record()
    #     out = self.lm(input_ids=input_ids,
    #                   position_ids=position_ids,
    #                   past_key_values=cache,
    #                   use_cache=True)
    #     end.record()
    #     torch.cuda.synchronize()
    #     response_time = start.elapsed_time(end)

    @torch.inference_mode()
    def run_latency_eval(self):

        for dataset_name in self.dataset_list:

            dataset = self.dataset_list[dataset_name]
            dataset.init(limit_entries=None)

            # create result directory
            device_used = "cpu" if self.use_cpu_for_inference else "gpu"
            cache_used = "cache" if self.enable_cache else "no_cache"
            result_path = os.path.join(BENCHMARK_PATH, "results_latency")
            no_cache = not self.enable_cache

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            results = []

            for entry in tqdm(dataset.entries):

                schema_file_path = os.path.join(SCHEMA_FILE_DIRECTORY, dataset_name, entry.schema)

                self.cache_engine.add_schema(read_file(schema_file_path, self.preproc), no_cache=no_cache,
                                             max_tokens=3500)

                prompt = Prompt(entry.prompt, self.preproc)

                token_ids, position_ids, cache_time, cache = self.cache_engine.process(prompt, no_cache=no_cache,
                                                                                       return_full_position_ids=self.lm.use_full_position_ids)

                input_ids = torch.tensor([token_ids], device=self.lm.device, dtype=torch.long)
                position_ids = torch.tensor([position_ids], device=self.lm.device, dtype=torch.long)
                # print(len(position_ids[0]))

                # add redundant batch dim
                if cache is not None:
                    cache = [(k[0].unsqueeze(0), k[1].unsqueeze(0)) for k in cache]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                out = self.lm(input_ids=input_ids,
                              position_ids=position_ids,
                              past_key_values=cache,
                              use_cache=True)
                end.record()
                torch.cuda.synchronize()
                response_time = start.elapsed_time(end)

                result = {
                    "entry_schema": entry.schema,
                    "cache_time": cache_time,
                    "response_time": response_time,
                }
                # print(result)
                results.append(result)

                self.cache_engine.remove_all_schemas()

            with open(os.path.join(result_path, f"{self.model_name}-{device_used}-{cache_used}-{dataset_name}.json"),
                      "w") as f:
                json.dump(
                    {
                        'model_name': self.model_name,
                        'device_used': device_used,
                        'cache_used': cache_used,
                        'dataset_name': dataset_name,

                        'results': results
                    }, f)
                f.write("\n")


def main(llm_config_path: str = os.path.join('./', "config/llm_config_llama2.json"),
         enable_cache=True,
         use_cpu_for_inference=False):
    eval = Eval(llm_config_path, enable_cache, use_cpu_for_inference)

    eval.run_latency_eval()


if __name__ == "__main__":
    fire.Fire(main)
