import torch.cuda
import fire
import sys, json
import os
import datetime

from benchmark.longbench import LongBench
from promptcache.model import Llama2, Falcon, Mpt
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters

from benchmark.benchmark_base import DATASET_LIST, SCHEMA_FILE_DIRECTORY
from benchmark.squad_v2 import SquadV2
from benchmark.multi_news import MultiNews
from benchmark.ms_marco_v1_1 import MSMarcoV1

BENCHMARK_PATH = "./benchmark"


class Eval:
    def __init__(self, llm_config_path, dataset, enable_cache):
        with open("./config/dataset_maxlen.json", 'r') as f:
            self.dataset_maxlen = json.load(f)

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

        # self.parameter = GenerationParameters(
        #     temperature=0.1,
        #     repetition_penalty=1.17,
        #     top_p=0.95,
        #     top_k=-1,
        #     max_new_tokens=512,
        #     stop_token_ids=self.lm.stop_token_ids,
        #     stop_str=self.lm.stop_str
        # )

        self.parameter = GenerationParameters(
            temperature=0.0,
            repetition_penalty=1.17,
            top_p=-1,
            top_k=1,
            max_new_tokens=self.dataset_maxlen[dataset],
            stop_token_ids=self.lm.stop_token_ids,
            stop_str=self.lm.stop_str
        )

        if dataset is None or dataset not in DATASET_LIST:
            raise ValueError("Dataset name cannot be None, valid dataset names are: " + ", ".join(DATASET_LIST))

        match dataset:
            case "squad_v2":
                self.dataset = SquadV2()

            case "multi_news":
                self.dataset = MultiNews()

            case "ms_marco":
                self.dataset = MSMarcoV1()

            case "narrativeqa":
                self.dataset = LongBench("narrativeqa")

            case "qasper":
                self.dataset = LongBench("qasper")

            case "multifieldqa_en":
                self.dataset = LongBench("multifieldqa_en")

            case "hotpotqa":
                self.dataset = LongBench("hotpotqa")

            case "2wikimqa":
                self.dataset = LongBench("2wikimqa")

            case "musique":
                self.dataset = LongBench("musique")

            case "dureader":
                self.dataset = LongBench("dureader")

            case "gov_report":
                self.dataset = LongBench("gov_report")

            case "qmsum":
                self.dataset = LongBench("qmsum")

            case "multi_news_long":
                self.dataset = LongBench("multi_news")

            case "vcsum":
                self.dataset = LongBench("vcsum")

            case "trec":
                self.dataset = LongBench("trec")

            case "triviaqa":
                self.dataset = LongBench("triviaqa")

            case "samsum":
                self.dataset = LongBench("samsum")

            case "lsht":
                self.dataset = LongBench("lsht")

            case "passage_count":
                self.dataset = LongBench("passage_count")

            case "passage_retrieval_en":
                self.dataset = LongBench("passage_retrieval_en")

            case "lcc":
                self.dataset = LongBench("lcc")

            case "repobench-p":
                self.dataset = LongBench("repobench-p")

        # for testing purpose, limit the entries to a small number
        self.dataset.init(limit_entries=10)

        # create result directory
        self.result_directory = os.path.join(BENCHMARK_PATH, "results",
                                             f"{self.model_name}-{self.dataset.dataset_name}",
                                             datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
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

    @torch.inference_mode()
    def run_latency_eval(self):

        for entry in self.dataset.entries:

            schema_file_path = os.path.join(SCHEMA_FILE_DIRECTORY, self.dataset.dataset_name, entry.schema)
            print(schema_file_path)
            if True:
                self.cache_engine.add_schema(read_file(schema_file_path, self.preproc))

            prompt = Prompt(entry.prompt, self.preproc)

            no_cache = not self.enable_cache

            token_ids, position_ids, cache_time, cache = self.cache_engine.process(prompt, no_cache=no_cache,
                                                                                   return_full_position_ids=self.lm.use_full_position_ids)

            if no_cache:
                assert cache is None

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
                "cache_time": cache_time,
                "response_time": response_time,
            }
            print(result)
            self.store_results(result)

            self.cache_engine.remove_all_schemas()

    def run(self, cache_batch_size, split):
        entry_count = self.dataset.get_entry_count()
        split_count = entry_count // split[1]

        start = split_count * split[0]
        end = split_count * (split[0] + 1)
        print(f"Running benchmark on {self.dataset.dataset_name}, start: {start}, end: {end}")

        for i in range(start, end, cache_batch_size):
            entries = self.dataset.get_query((i, i + cache_batch_size))
            # load schema for `cache_batch_size` entries
            for entry in entries:
                schema_file_path = os.path.join(SCHEMA_FILE_DIRECTORY, self.dataset.dataset_name, entry.schema)
                print(schema_file_path)
                self.cache_engine.add_schema(read_file(schema_file_path, self.preproc),
                                             batch_size=self.llm_config.get("schema_load_batch", 1))

            for entry in entries:
                prompt = Prompt(entry.prompt, self.preproc)
                print(entry.prompt)
                no_cache = not self.enable_cache
                token_ids, position_ids, cache_time, cache = self.cache_engine.process(prompt, no_cache=no_cache,
                                                                                       return_full_position_ids=self.lm.use_full_position_ids)
                if no_cache:
                    assert cache is None

                output_stream = self.gen_engine.generate(token_ids, position_ids, self.parameter, cache,
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
                        print(tt, end=" ", flush=True)
                        pre = now

                result = {
                    "cache_time": cache_time,
                    "response_time": response_time,
                    "answers": entry.answer,
                    "response": resp
                }
                self.store_results(result)
                print("\n")

            self.cache_engine.remove_all_schemas()


def main(llm_config_path: str = os.path.join('./', "config/llm_config_llama2.json"),
         dataset: str = "2wikimqa", enable_cache=True, cache_batch_size=1, split=(0, 1), test_latency=False):
    eval = Eval(llm_config_path, dataset, enable_cache)

    if test_latency:
        eval.run_latency_eval()
    else:
        eval.run(cache_batch_size, split)


if __name__ == "__main__":
    fire.Fire(main)
