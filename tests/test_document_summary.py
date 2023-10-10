# Add the parent directory to the sys.path list
import os, sys
document_summary_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(document_summary_path, '..')))

import unittest
from benchmark.multi_news import MultiNews

## The following code is for testing the MultiNews class
class TestMultiNews(unittest.TestCase):
    def setUp(self):
        self.document_summary = MultiNews()
        self.document_summary.init(verbose=False)

    def test_get_next_query(self):
        query_id, query, modules = self.document_summary.get_next_query()
        self.assertIsInstance(query_id, int)
        self.assertGreaterEqual(query_id, 0)
        self.assertIsInstance(query, str)
        self.assertEqual(query, "")    # query is empty string for summary benchmark
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
