import json
import os
import sys


class JsonParser:
    def __init__(self, file_path):
        # add this directory to the sys.path list
        # print(sys.path)
        print(f'Parsing {file_path}...')
        assert os.path.isfile(file_path)
        self.file_path = file_path
        self.data = None

    def parse(self):
        with open(self.file_path) as f:
            self.data = json.load(f)

    def get_data(self):
        return self.data


class BenchmarkSetupParser(JsonParser):
    dataset_size_str = 'dataset_sizes'

    def __init__(self, file_path):
        super().__init__(file_path)

    def parse(self):
        super().parse()

    def get_data_size(self, size_name: str):
        assert self.data is not None
        assert size_name in self.data[BenchmarkSetupParser.dataset_size_str]
        return int(self.data[BenchmarkSetupParser.dataset_size_str][size_name])


class BenchmarkProfileParser(JsonParser):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.benchmark_setup_parser: BenchmarkSetupParser = None

    def parse(self):
        super().parse()
        # try to get the benchmark setup
        assert 'benchmark_dataset_name' in self.data
        setup_json_path = f'benchmark/{self.data["benchmark_dataset_name"]}/setup.json'
        self.benchmark_setup_parser = BenchmarkSetupParser(setup_json_path)
        self.benchmark_setup_parser.parse()
        print(f'Parsing benchmark setup from {setup_json_path}...')
        # print(f'Dataset size: {self.benchmark_setup_parser.get_data_size(self.data["dataset_size"])}')

    def get_benchmark_name(self):
        return self.data['benchmark_name']

    def get_benchmark_description(self):
        return self.data['benchmark_description']

    def get_benchmark_dataset_name(self):
        return self.data['benchmark_dataset_name']

    def get_benchmark_dataset_comment(self):
        return self.data['benchmark_dataset_comment']

    def get_dataset_size(self):
        return self.data['dataset_size']

    def get_prompt_cache(self):
        return bool(self.data['prompt_cache'])
