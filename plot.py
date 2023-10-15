import os
import glob
import json

import matplotlib.pyplot as plt


def load_latency_profile(path):
    model_name = None
    device_used = None
    cache_used = None

    output = {}

    for json_file in glob.glob(path):
        with open(json_file, 'r') as f:
            data = json.load(f)

        if model_name is None:
            model_name = data["model_name"]
            device_used = data["device_used"]
            cache_used = data["cache_used"]

        # ensure correctness
        assert data["model_name"] == model_name
        assert data["device_used"] == device_used
        assert data["cache_used"] == cache_used

        dataset_name = data["dataset_name"]

        results = data["results"]

        avg_latency_lb = 0
        avg_latency_ub = 0

        for result in results:
            cache_time = float(result["cache_time"])
            response_time = float(result["response_time"])

            avg_latency_lb += response_time
            avg_latency_ub += response_time + cache_time

        avg_latency_lb /= len(results)
        avg_latency_ub /= len(results)

        output[dataset_name] = {
            "latency_lb": avg_latency_lb,
            "latency_ub": avg_latency_ub
        }

    return output


def plot_latency_gpu(model_name: str):
    # get all files
    path_gpu_no_cache = f"./benchmark/results_latency/{model_name}-gpu-no_cache-*.json"
    path_gpu_cache = f"./benchmark/results_latency/{model_name}-gpu-cache-*.json"

    result_gpu_no_cache = load_latency_profile(path_gpu_no_cache)
    result_gpu_cache = load_latency_profile(path_gpu_cache)

    print(result_gpu_no_cache)
    print(result_gpu_cache)




def main():
    plot_latency_gpu("llama")


if __name__ == '__main__':
    main()
