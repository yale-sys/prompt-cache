import sys
import os

# Add the parent directory to the sys.path list
document_summary_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(document_summary_path, '..')))

from benchmark_base import Benchmark
from dataset_download import load_documentation_summary

_document_header = "Document"
_document_dataset = "train"

class DocumentSummary(Benchmark):
    def __init__(self):
        super().__init__()
        self.next_query_idx = 0

    def init(self, verbose=False):
        """
        Download (one time) and load the dataset to run; 
        Do any preprocessing required for running this benchmark.
        """
        self.dataset = load_documentation_summary()
        if verbose:
            print("Dataset loaded. First entry below:")
            print(self.dataset[_document_dataset][0])

    def get_documents(self):
        """
        Return a list of paths for XML files that need to be cached.
        """
        return [document_summary_path + "/schema_summary_sample.xml", ]

    def get_next_query(self):
        """
        Return query_id (unsigned), query (string), and chosen modules (a list of string).
        """
        assert self.dataset is not None
        assert self.dataset[_document_dataset] is not None
        assert self.next_query_idx < len(self.dataset[_document_dataset])
        response = self.next_query_idx, "", [f"{_document_header}{self.next_query_idx}"]
        self.next_query_idx += 1
        return response
        
    def evaluate(self, query_id, response_from_llm):
        """
        Take query_id and response_from_llm as parameters and return a score in the range [0,1].
        """
        assert self.dataset is not None
        assert self.dataset[_document_dataset] is not None
        assert query_id < len(self.dataset[_document_dataset])
        assert response_from_llm is not None
        assert response_from_llm != ""
        raise NotImplementedError("This method should call utility function to measure how the response is closer to the expected answer.")

## The following code is for testing the DocumentSummary class
import unittest
import os
from document_summary import DocumentSummary

class TestDocumentSummary(unittest.TestCase):
    def setUp(self):
        self.document_summary = DocumentSummary()
        self.document_summary.init(verbose=False)

    def test_get_documents(self):
        doc_to_cache = self.document_summary.get_documents()
        for file in doc_to_cache:
            self.assertTrue(os.path.exists(file))

    def test_get_next_query(self):
        query_id, query, modules = self.document_summary.get_next_query()
        self.assertIsInstance(query_id, int)
        self.assertGreaterEqual(query_id, 0)
        self.assertIsInstance(query, str)
        self.assertEquals(query, "")    # query is empty string for summary benchmark
        self.assertIsInstance(modules, list)
        self.assertGreaterEqual(len(modules), 1)
        for module in modules:
            self.assertIsInstance(module, str)

    def test_evaluate(self):
        query_id, query, modules = self.document_summary.get_next_query()
        response = self.document_summary.evaluate(query_id, "response")
        self.assertIsInstance(response, float)
        self.assertGreaterEqual(response, 0.)
        self.assertLessEqual(response, 1.)

if __name__ == '__main__':
    unittest.main()
