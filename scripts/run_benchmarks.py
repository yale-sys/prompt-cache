import subprocess
import logging
import json
import os
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor

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
                command = "python3 eval.py"
                merged_args = {**default_args, **benchmark, "llm_config_path": llm_config}
                for key, value in merged_args.items():
                    if key == "enable_cache":
                        continue
                    command += f" --{key} {value}"
                command += f" --enable_cache {enable_cache}"
                command += f" --test_latency= False"
                command += f" --cache_batch_size 1"
                commands.append(command)
    return commands

def run_python_command_with_logging(command):
    try:
        # parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
        # os.chdir(parent_dir)
        # logging.info(f"Changed working directory to {parent_dir}")
        
        subprocess.run("cd .. && " + command, shell=True, check=True)
        logging.info(f"Command {command} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command {command} failed: {e}")
        return False

def main():
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(filename=f'benchmark_results_{datetime.date.today().strftime("%Y%m%d_%H%M%S")}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    num_gpus = detect_nvidia_gpus()
    logging.info(f"Detected {num_gpus} Nvidia GPUs.")
    
    # Read arguments from JSON file
    args_dict = read_args_from_json("benchmark_setup.json")
    
    # Construct the Python commands
    python_commands = construct_python_commands(args_dict["default"], args_dict["benchmarks"], args_dict["llm_list"])
    logging.info(f"Constructed {len(python_commands)} benchmarks.")
    
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        results = list(executor.map(run_python_command_with_logging, python_commands))

if __name__ == "__main__":
    main()
