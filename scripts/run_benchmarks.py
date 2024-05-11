import subprocess
import logging
import json
import os
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import queue


def detect_nvidia_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        num_gpus = len(result.stdout.strip().split('\n'))
        return num_gpus
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 0

def read_args_from_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def construct_python_commands(default_args, benchmarks, llm_list):
    commands = []
    for llm in llm_list:
        _llm_name = llm["llm"]
        llm_config = llm["config_name"]
        for benchmark in benchmarks:
            for enable_cache in [True, False]:
                split = benchmark.get("split", 1)
                for index in range(split):
                    command = "python3 eval.py"
                    merged_args = {**default_args, **benchmark, "llm_config_path": llm_config}
                    for key, value in merged_args.items():
                        if key == "enable_cache":
                            continue
                        elif key == "split":
                            command += f" --split {index},{split}"
                        else:
                            command += f" --{key} {value}"
                    command += f" --enable_cache {enable_cache}"
                    command += f" --test_latency False"
                    commands.append(command)
    return commands
    
global python_commands_list
def gpu_worker(gpu_id, command_lock):
    while True:
        with command_lock:
            global next_command_index
            if next_command_index >= len(python_commands_list):
                return  # All commands are executed, so exit the thread
            command = python_commands_list[next_command_index]
            next_command_index += 1
        
        env_command = f"CUDA_VISIBLE_DEVICES={gpu_id} " + command
        try:
            subprocess.run("cd .. && " + env_command, shell=True, check=True)
            logging.info(f"Worker using GPU {gpu_id}: Command {command} completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Worker using GPU {gpu_id}: Command {command} failed: {e}")


def main():
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(filename=f'benchmark_results_{datetime.date.today().strftime("%Y%m%d_%H%M%S")}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    num_gpus = detect_nvidia_gpus()
    logging.info(f"Detected {num_gpus} Nvidia GPUs.")
    
    # Read arguments from JSON file
    args_dict = read_args_from_json("benchmark_setup.json")
    
    global python_commands_list
    # Construct the Python commands
    python_commands_list = construct_python_commands(args_dict["default"], args_dict["benchmarks"], args_dict["llm_list"])
    logging.info(f"Constructed {len(python_commands_list)} benchmarks.")

    global next_command_index
    next_command_index = 0
    command_lock = threading.Lock()
    # Start a thread for each GPU
    threads = []
    for gpu_id in range(num_gpus):
        t = threading.Thread(target=gpu_worker, args=(gpu_id, command_lock))
        t.start()
        threads.append(t)
    
    # Wait for all threads to finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
