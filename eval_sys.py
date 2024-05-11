import gc

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
from torch.profiler import profile, record_function, ProfilerActivity


class Eval:
    def __init__(self, memo, llm_config_path, use_cpu_for_inference=False):
        with open("./config/dataset_maxlen.json", 'r') as f:
            self.dataset_maxlen = json.load(f)

        with open(llm_config_path, 'r') as f:
            self.llm_config = json.load(f)
        self.memo = memo
        self.use_cpu_for_inference = use_cpu_for_inference
        self.repeat_times = 2 if use_cpu_for_inference else 3
        self.model_name = self.llm_config["name"]
        self.model_arch = self.llm_config["arch"]
        self.model_log_name = self.llm_config["log_name"]
        self.max_ctx_length = self.llm_config.get("max_ctx_length", 4096)
        self.max_tokens = self.llm_config.get("max_tokens", 3500)

        if self.model_arch == "llama":
            self.lm_for_caching = Llama2(name=self.model_name, device_map={"": 0}, load_in_8bit=True)
        elif self.model_arch == "falcon":
            self.lm_for_caching = Falcon(name=self.model_name, device_map={"": 0}, load_in_8bit=True)
        elif self.model_arch == "mpt":
            self.lm_for_caching = Mpt(name=self.model_name, device_map={"": 0}, load_in_8bit=True)
        else:
            raise ValueError("Invalid model name")

        if self.use_cpu_for_inference:
            if self.model_arch == "llama":
                self.lm = Llama2(name=self.model_name, device_map=None)
            elif self.model_arch == "falcon":
                self.lm = Falcon(name=self.model_name, device_map=None)
            elif self.model_arch == "mpt":
                self.lm = Mpt(name=self.model_name, device_map=None)
        else:
            self.lm = self.lm_for_caching

        self.cache_engine = CacheEngine(self.max_ctx_length, self.lm,
                                        target_device=self.lm.device)
        self.gen_engine = GenerationEngine(self.lm)
        self.preproc = [
            # CompactSpaces(),
            self.lm.get_formatter()
        ]

        self.dataset_list = {
            "narrativeqa": LongBench("narrativeqa"),
            "qasper": LongBench("qasper"),
            "multifieldqa_en": LongBench("multifieldqa_en"),
            "hotpotqa": LongBench("hotpotqa"),
            "2wikimqa": LongBench("2wikimqa"),
            "musique": LongBench("musique"),
            "gov_report": LongBench("gov_report"),
            "qmsum": LongBench("qmsum"),
            "multi_news": LongBench("multi_news"),
            "triviaqa": LongBench("triviaqa"),
            "samsum": LongBench("samsum"),
            "passage_count": LongBench("passage_count"),
            "passage_retrieval_en": LongBench("passage_retrieval_en"),
        }

    # @torch.inference_mode()
    # def profile_cpu_inference(self):
    #
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #         with record_function("model_inference"):
    #             model(inputs)

    # recomputation overhead vs mem trasnfer overhead
    @torch.inference_mode()
    def run_critical_point(self):

        def create_cache(seq_len):

            # # llama 2 13B
            num_layers = 40
            num_heads = 40
            head_dim = 128

            # # llama 2 7B
            # num_layers = 32
            # num_heads = 32
            # head_dim = 128

            return [(torch.rand((num_heads, seq_len, head_dim), dtype=torch.float16, device='cpu'),
                     torch.rand((num_heads, seq_len, head_dim), dtype=torch.float16, device='cpu')) for _ in
                    range(num_layers)]

        test_seq_len = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            512 + 128 * 1,
            512 + 128 * 2,
            512 + 128 * 3,
            1024,
            1024 + 256 * 1,
            1024 + 256 * 2,
            1024 + 256 * 3,
            2048,
            2028 + 512 * 1,
            2028 + 512 * 2,
            2028 + 512 * 3,
            4096,
            # 4096 + 1024 * 1,
            # 4096 + 1024 * 2,

        ]

        results = []

        for seq_len in tqdm(test_seq_len):
            for _ in range(self.repeat_times):
                ## 1. compute gpu upload time
                kv_cache = create_cache(seq_len)

                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                # upload everything to GPU
                kv_cache_gpu = [
                    (k[0].to('cuda', non_blocking=True, copy=True), k[1].to('cuda', non_blocking=True, copy=True))
                    for k in kv_cache]

                end.record()
                torch.cuda.synchronize()
                gpu_upload_time = start.elapsed_time(end)

                del kv_cache_gpu, kv_cache
                gc.collect()
                torch.cuda.empty_cache()

                results.append({
                    "seq_len": seq_len,
                    "time": gpu_upload_time,
                })

        result_path = os.path.join(BENCHMARK_PATH, "results_latency")

        with open(os.path.join(result_path, f"{self.memo}-{self.model_log_name}-critical_point-upload.json"),
                  "w") as f:
            json.dump(
                {
                    'model_name': self.model_name,
                    'results': results
                }, f)

        results = []
        ## 2. compute recomputation time
        for seq_len in tqdm(test_seq_len):

            for _ in range(self.repeat_times):
                token_ids = [100] * seq_len
                position_ids = list(range(seq_len))

                input_ids = torch.tensor([token_ids], device=self.lm.device, dtype=torch.long)
                position_ids = torch.tensor([position_ids], device=self.lm.device, dtype=torch.long)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                out = self.lm(input_ids=input_ids,
                              position_ids=position_ids,
                              past_key_values=None,
                              use_cache=False)

                end.record()
                torch.cuda.synchronize()
                recomputation_time = start.elapsed_time(end)

                del out
                gc.collect()
                torch.cuda.empty_cache()

                results.append({
                    "seq_len": seq_len,
                    "time": recomputation_time
                })

        result_path = os.path.join(BENCHMARK_PATH, "results_latency")

        with open(os.path.join(result_path, f"{self.memo}-{self.model_log_name}-critical_point-recomputation.json"),
                  "w") as f:
            json.dump(
                {
                    'model_name': self.model_log_name,
                    'results': results
                }, f)

    @torch.inference_mode()
    def run_critical_point22(self):


        test_seq_len = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            512 + 128 * 1,
            512 + 128 * 2,
            512 + 128 * 3,
            1024,
            1024 + 256 * 1,
            1024 + 256 * 2,
            1024 + 256 * 3,
            2048,
            2028 + 512 * 1,
            2028 + 512 * 2,
            2028 + 512 * 3,
            #4096,
            # 4096 + 1024 * 1,
            # 4096 + 1024 * 2,

        ]

        results = []

        for seq_len in tqdm(test_seq_len):
            for _ in range(self.repeat_times):
                ## 1. compute gpu upload time
                input_ids = torch.tensor([[100]], device=self.lm.device, dtype=torch.long)
                #position_ids = torch.tensor([[100]], device=self.lm.device, dtype=torch.long)

                device_cache = [
                    (torch.empty(1, 32, seq_len, 128, device=self.lm.device, dtype=torch.half),  # key
                     torch.empty(1, 32, seq_len, 128, device=self.lm.device, dtype=torch.half)) for _ in
                    range(32)]

                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                # upload everything to GPU
                out = self.lm(input_ids=input_ids,
                              #position_ids=position_ids,
                              past_key_values=device_cache,
                              use_cache=True)

                end.record()
                torch.cuda.synchronize()
                gpu_upload_time = start.elapsed_time(end)

                del device_cache
                gc.collect()
                torch.cuda.empty_cache()

                results.append({
                    "seq_len": seq_len,
                    "time": gpu_upload_time,
                })

        result_path = os.path.join(BENCHMARK_PATH, "aaa")

        with open(os.path.join(result_path, f"{self.memo}-{self.model_log_name}-critical_point-upload.json"),
                  "w") as f:
            json.dump(
                {
                    'model_name': self.model_name,
                    'results': results
                }, f)

        results = []


    @torch.inference_mode()
    def run_latency_eval(self, do_cache):

        for dataset_name in self.dataset_list:

            dataset = self.dataset_list[dataset_name]
            dataset.init(limit_entries=5)

            # create result directory
            device_used = "cpu" if self.use_cpu_for_inference else "gpu"
            cache_used = "cache" if do_cache else "no_cache"
            result_path = os.path.join(BENCHMARK_PATH, "results_latency")
            no_cache = not do_cache

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            results = []

            for entry in tqdm(dataset.entries[:5]):
                for _ in range(self.repeat_times):
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

                    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    #     with record_function("model_inference"):
                    out = self.lm(input_ids=input_ids,
                                  position_ids=position_ids,
                                  past_key_values=cache,
                                  use_cache=True)
                    end.record()
                    torch.cuda.synchronize()
                    response_time = start.elapsed_time(end)

                    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                    result = {
                        "entry_schema": entry.schema,
                        "cache_time": cache_time,
                        "response_time": response_time,
                    }
                    # print(result)
                    results.append(result)

                    self.cache_engine.remove_all_schemas()

            with open(
                    os.path.join(result_path,
                                 f"{self.memo}-{self.model_log_name}-{device_used}-{cache_used}-{dataset_name}.json"),
                    "w") as f:
                json.dump(
                    {
                        'model_name': self.model_log_name,
                        'device_used': device_used,
                        'cache_used': cache_used,
                        'dataset_name': dataset_name,

                        'results': results
                    }, f)
                f.write("\n")

    @torch.inference_mode()
    def run_profile(self, do_cache):
        device_used = "cpu" if self.use_cpu_for_inference else "gpu"
        cache_used = "cache" if do_cache else "no_cache"

        for dataset_name in self.dataset_list:

            dataset = self.dataset_list[dataset_name]
            dataset.init(limit_entries=5)

            no_cache = not do_cache

            for entry in tqdm(dataset.entries[:5]):
                for _ in range(self.repeat_times):
                    schema_file_path = os.path.join(SCHEMA_FILE_DIRECTORY, dataset_name, entry.schema)

                    self.cache_engine.add_schema(read_file(schema_file_path, self.preproc), no_cache=no_cache,
                                                 max_tokens=2500)

                    prompt = Prompt(entry.prompt, self.preproc)

                    token_ids, position_ids, cache_time, cache = self.cache_engine.process(prompt,
                                                                                           no_cache=no_cache,
                                                                                           return_full_position_ids=self.lm.use_full_position_ids)

                    input_ids = torch.tensor([token_ids], device=self.lm.device, dtype=torch.long)
                    position_ids = torch.tensor([position_ids], device=self.lm.device, dtype=torch.long)
                    # print(len(position_ids[0]))

                    # add redundant batch dim
                    if cache is not None:
                        cache = [(k[0].unsqueeze(0), k[1].unsqueeze(0)) for k in cache]

                    with profile(activities=[ProfilerActivity.CUDA], with_stack=True,
                                 experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
                        with record_function("model_inference"):
                            out = self.lm(input_ids=input_ids,
                                          position_ids=position_ids,
                                          past_key_values=cache,
                                          use_cache=True)

                    prof.export_stacks(f"./profile/{device_used}_{cache_used}_self_cuda_time_total.txt",
                                       "self_cuda_time_total")
                    self.cache_engine.remove_all_schemas()

                    return


def main(memo: str = "13900k-cpu", llm_config_path: str = os.path.join('./', "config/llm_config_llama2_7b.json"),
         use_cpu_for_inference=True):
    eval = Eval(memo, llm_config_path, use_cpu_for_inference)

    # eval.run_latency_eval(False)
    # eval.run_latency_eval(True)
    #eval.run_profile(True)
    #eval.run_profile(False)

    eval.run_critical_point22()


if __name__ == "__main__":
    fire.Fire(main)
