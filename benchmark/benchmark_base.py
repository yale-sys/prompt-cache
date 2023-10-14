# It provides APIs to load/initialize the dataset and evaluate the response from the llm

# * Required API
#     * init() : download (one time) and load the dataset to run; do any preprocessing required for running this benchmark
#     * get_entry_count(): return the number of entries in the dataset.
#     * get_query(): return a list of Entry objects for the given range.
import os, json
import abc
from typing import Tuple, List

SCHEMA_FILE_DIRECTORY = "./benchmark/schema"
DATASET_LIST = ["squad_v2", "multi_news", "wiki_qa", "pubmed_qa", "ms_marco", "narrativeqa", "qasper",
                "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news_long",
                "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "lcc",
                "repobench-p"]

DATASET_SUBSET = {
    "pubmed_qa": ["pqa_artificial", "pqa_labeled", "pqa_unlabeled"],
    "ms_marco": ["v1.1", "v2.1"]
}

class Entry:
    def __init__(self, schema, prompt, answer=None):
        """
        Constructor to initialize any required variables.
        [schema: str] path to the schema file, usage: cache_engine.add_schema(read_file(schema, preproc))
        [prompt: str] prompt text, which I should feed to the llm directly, it contains the used schema name and the question from dataset
        [answer: [str]] the potential answer list to the above question
        """
        self.schema = schema
        self.prompt = prompt
        self.answer = answer

    def __repr__(self) -> str:
        return f"Entry(schema={self.schema}, prompt={self.prompt}, answer={self.answer})"


class Benchmark(abc.ABC):
    def __init__(self, dataset_name: str):
        """
        Constructor to initialize any required variables.
        """
        if dataset_name not in DATASET_LIST:
            raise ValueError("Dataset name cannot be None, valid dataset names are: " + ", ".join(DATASET_LIST))

        self.dataset_name = dataset_name
        self.dataset = None
        self.entries = []
        self.schema_path = os.path.join(SCHEMA_FILE_DIRECTORY, dataset_name)
        if not os.path.exists(self.schema_path):
            os.makedirs(self.schema_path)

        self.load_prompt()

    def load_prompt(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_prompt_path = os.path.join(base_path, '../config/dataset_prompt.json')
        with open(dataset_prompt_path, 'r') as f:
            self.dataset_prompt = json.load(f)
        self.dataset_prompt = self.dataset_prompt[self.dataset_name]

    @abc.abstractmethod
    def init(self, limit_entries=None):
        """
        Download (one time) and load the dataset to run; 
        Preprocess the dataset to be organized in the `Entry` format.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def get_entry_count(self) -> int:
        """
        Return the number of entries in the dataset.
        """
        return len(self.entries)

    def get_query(self, range: Tuple[int, int]) -> List[Entry]:
        """
        Return a list of Entry objects for the given range.
        [range: (int, int)] the range of entries to return
        """
        return self.entries[range[0]:range[1]]
