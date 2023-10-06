# It provides APIs to load/initialize the dataset and evaluate the response from the llm

# * Required API
#     * init() : download (one time) and load the dataset to run; do any preprocessing required for running this benchmark
#     * get_documents(): return a list of the paths that need to be cached (the xml files)
#     * get_next_query(): return query id (unsigned), query (string), and chosen modules (a list of string)
#     * evaluate(query_id, response_from_llm): return score in [0,1] per query

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
        Do any preprocessing required for running this benchmark.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def get_documents(self) -> list:
        """
        Return a list of paths for XML files that need to be cached.
        """
        raise NotImplementedError("This method should be overridden by subclass")

    def get_next_query(self) -> (int, str, list):
        """
        Return query_id (unsigned), query (string), and chosen modules (a list of string).
        """
        raise NotImplementedError("This method should be overridden by subclass")
        
    def evaluate(self, query_id: int, response_from_llm: str) -> float:
        """
        Take query_id and response_from_llm as parameters and return a score in the range [0,1].
        """
        raise NotImplementedError("This method should be overridden by subclass")

# Test the base class by creating a subclass and overriding the methods
class ExampleBenchmark(Benchmark):
    def init(self):
        print("Example init")

    def get_documents(self):
        return ["path1.xml", "path2.xml"]
    
    def get_next_query(self):
        return 1, "query_string", ["module1", "module2"]
    
    def evaluate(self, query_id, response_from_llm):
        return 0.9

# Create an instance of the ExampleBenchmark and test the methods
if __name__ == "__main__":
    example_benchmark = ExampleBenchmark()
    example_benchmark.init()
    print(example_benchmark.get_documents())
    query_id, query, modules = example_benchmark.get_next_query()
    print(query_id, query, modules)
    print(example_benchmark.evaluate(query_id, "response"))
    pass
