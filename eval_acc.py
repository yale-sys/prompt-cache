import math
import random

import numpy as np
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

from benchmark.benchmark_base import DATASET_LIST, SCHEMA_FILE_DIRECTORY
from benchmark.squad_v2 import SquadV2
from benchmark.multi_news import MultiNews
from promptcache.prompt import apply_preproc
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score
)
from multiprocessing import cpu_count, Process, Queue
from concurrent.futures import ProcessPoolExecutor

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

BENCHMARK_PATH = "./benchmark"


class Eval:
    def __init__(self, gpu_id, llm_config_path, dataset_list, enable_cache):
        with open("./config/dataset_maxlen.json", 'r') as f:
            self.dataset_maxlen = json.load(f)

        with open(llm_config_path, 'r') as f:
            self.llm_config = json.load(f)

        self.enable_cache = enable_cache

        self.model_name = self.llm_config["name"]
        self.model_arch = self.llm_config["arch"]
        self.model_log_name = self.llm_config["log_name"]
        self.max_ctx_length = self.llm_config.get("max_ctx_length", 4096)
        self.max_tokens = self.llm_config.get("max_tokens", 3500)
        self.dataset_list = dataset_list

        if self.model_arch == "llama":
            self.lm = Llama2(name=self.model_name, device_map={"": gpu_id}, load_in_8bit=True)
        elif self.model_arch == "falcon":
            self.lm = Falcon(name=self.model_name, device_map={"": gpu_id}, load_in_8bit=True)
        elif self.model_arch == "mpt":
            self.lm = Mpt(name=self.model_name, device_map={"": gpu_id}, load_in_8bit=True)
        else:
            raise ValueError("Invalid model name")

        self.cache_engine = CacheEngine(self.max_ctx_length, self.lm,
                                        target_device=self.lm.device)
        self.gen_engine = GenerationEngine(self.lm)
        self.preproc = [
            # CompactSpaces(),
            self.lm.get_formatter()
        ]

        # create result directory
        self.result_directory = os.path.join(BENCHMARK_PATH, "results_acc")
        if not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory)

    def run(self):

        for dataset_name in self.dataset_list:

            dataset = self.dataset_list[dataset_name]
            dataset.init(limit_entries=3)

            results = []

            for entry in tqdm(dataset.entries):
                # print(entry.prompt)

                schema = apply_preproc(entry.schema, self.preproc)
                prompt = Prompt(entry.prompt, self.preproc)

                self.cache_engine.add_schema(schema, max_tokens=self.max_tokens)

                no_cache = not self.enable_cache
                token_ids, position_ids, cache_time, cache = self.cache_engine.process(prompt,
                                                                                       no_cache=no_cache,
                                                                                       return_full_position_ids=self.lm.use_full_position_ids)
                if no_cache:
                    assert cache is None

                parameter = GenerationParameters(
                    temperature=0.0,
                    repetition_penalty=1.0,
                    top_p=0.0,
                    top_k=-1,
                    max_new_tokens=self.dataset_maxlen[dataset_name],
                    stop_token_ids=self.lm.stop_token_ids,
                    stop_str=self.lm.stop_str
                )

                output_stream = self.gen_engine.generate(token_ids, position_ids, parameter, cache,
                                                         stream_interval=2,
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
                        # print(tt, end=" ", flush=True)
                        pre = now

                tt = " ".join(output_text[pre:])
                # print(tt, flush=True)
                resp += tt
                # print("\n")

                result = {
                    "cache_time": cache_time,
                    "response_time": response_time,
                    "answers": entry.answer,
                    "response": resp
                }
                print(result)

                results.append(result)
                self.cache_engine.remove_all_schemas()

            total_score = 0
            metric_fn = dataset2metric[dataset_name]
            for result in results:
                response = result["response"]
                answers = result["answers"]

                score = 0.
                for answer in answers:
                    score = max(score, metric_fn(response, answer))

                total_score += score

            total_score = total_score / len(results) * 100
            print(f"Total score: {total_score:.2f}")

            if self.enable_cache:
                prefix = "cache_enabled"
            else:
                prefix = "cache_disabled"
            filename = f"{self.model_log_name}-{dataset_name}-{prefix}.json"

            with open(os.path.join(self.result_directory, filename), "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "model_arch": self.model_arch,
                    "dataset_name": dataset_name,
                    "enable_cache": self.enable_cache,
                    "total_score": total_score,
                    "results": results
                }, f)


def run_eval(gpu_id, llm_config_path: str = os.path.join('./', "config/llm_config_llama2_7b.json"),
             dataset: str = "narrativeqa",
             enable_cache=True, ):
    seed_everything(42)

    eval = Eval(gpu_id, llm_config_path, dataset, enable_cache)
    eval.run()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main(num_gpus=1, llm_config_path: str = os.path.join('./', "config/llm_config_llama2_7b.json"),
         enable_cache=True,
         ):
    dataset_list = {
        "narrativeqa": LongBench("narrativeqa"),
        "qasper": LongBench("qasper"),
        "multifieldqa_en": LongBench("multifieldqa_en"),
        "hotpotqa": LongBench("hotpotqa"),
        "2wikimqa": LongBench("2wikimqa"),
        "musique": LongBench("musique"),
        "gov_report": LongBench("gov_report"),
        "qmsum": LongBench("qmsum"),
        "multi_news": LongBench("multi_news"),
        "trec": LongBench("trec"),
        "triviaqa": LongBench("triviaqa"),
        "samsum": LongBench("samsum"),
        "passage_count": LongBench("passage_count"),
        "passage_retrieval_en": LongBench("passage_retrieval_en"),
        "lcc": LongBench("lcc"),
        "repobench-p": LongBench("repobench-p")
    }

    dpg = int(math.ceil(len(dataset_list) / num_gpus))

    jobs_list = []
    nn = list(dataset_list.keys())
    for i in range(num_gpus):
        dataset_names = nn[i * dpg:(i + 1) * dpg]
        jobs = {}
        for dn in dataset_names:
            jobs[dn] = dataset_list[dn]
        jobs_list.append(jobs)

    processes = [
        Process(target=run_eval, args=(i, llm_config_path, jobs_list[i], enable_cache))
        for i in range(num_gpus)
    ]

    for p in processes:
        p.start()

    seed_everything(42)

    for p in processes:
        p.join()


if __name__ == "__main__":
    fire.Fire(main)
