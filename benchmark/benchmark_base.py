# It provides APIs to load/initialize the dataset and evaluate the response from the llm

# * Required API
#     * init() : download (one time) and load the dataset to run; do any preprocessing required for running this benchmark
#     * get_entry_count(): return the number of entries in the dataset.
#     * get_query(): return a list of Entry objects for the given range.

class Entry:
    def __init__(self, schema, prompt, answer=None):
        """
        Constructor to initialize any required variables.
        [schema: str] path to the schema file, usage: cache_engine.add_schema(read_file(schema, preproc))
        [prompt: str] prompt text, which I should feed to the llm directly, it contains the used schema name and the question from dataset
        [answer: str] the answer to the above question
        """
        self.schema = schema
        self.prompt = prompt
        self.answer = answer

class Benchmark:
    def __init__(self):
        """
        Constructor to initialize any required variables.
        """
        self.dataset = None
        pass

    def init(self):
        """
        Download (one time) and load the dataset to run; 
        Preprocess the dataset to be organized in the `Entry` format.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def get_entry_count(self):
        """
        Return the number of entries in the dataset.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def get_query(self, range) -> [Entry]:
        """
        Return a list of Entry objects for the given range.
        [range: (int, int)] the range of entries to return
        """
        raise NotImplementedError("This method should be overridden by subclass")

