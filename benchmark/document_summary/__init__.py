import sys
import os

# Add the parent directory to the sys.path list
document_summary_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(document_summary_path, '..')))

from benchmark_base import Benchmark
from dataset_download import load_documentation_summary
from utils import XMLSchemaBuilder

_document_schema_name = "document_summary"
_document_header = "Document"
_document_dataset = "train"
_document_system_description = "Dialogues between a user and an AI about the document provided by the user with the aim of being helpful, aware, and accurate."
_document_user_description = "Summarize the following document in around five sentences:"
_document_assistant_description = "Sure. I have read the document. give me any instructions regarding summarization, and I will try to follow them."

MAX_DOCUMENT_LENGTH = 2560

class DocumentSummary(Benchmark):
    def __init__(self):
        super().__init__()
        self.next_query_idx = 0

    def init(self, xml_path=os.path.join(document_summary_path, "./schema_summary_sample.xml"), verbose=False):
        """
        Download (one time) and load the dataset to run; 
        Do any preprocessing required for running this benchmark.
        """
        self.dataset = load_documentation_summary()
        if verbose:
            print("Dataset loaded. First entry below:")
            print(self.dataset[_document_dataset][1])
        # Now we can generate xml files
        assert self.dataset is not None
        self._generate_xml(xml_path)
    
    def _generate_xml(self, xml_path, num_queries=10):
        # Generate xml files
        # Create an instance of XMLSchemaBuilder with the schema name "document_summary"
        builder = XMLSchemaBuilder(_document_schema_name)

        # Set the system description
        builder.set_system_description(_document_system_description)

        # Set the user description
        builder.set_user_description("")    # _document_user_description

        # Add document modules
        builder.add_document_module("DOC", "The given documents are the target for the summarization task. They can contain multiple sentences.")
        for i in range(num_queries):
            document_str = self.dataset[_document_dataset][i]["document"].replace("’", "'").replace("”", '"').replace("“", '"').replace("‘", "'").replace("…", "...").replace("–", "-")
            if len(document_str) > MAX_DOCUMENT_LENGTH:
                document_str = document_str[:MAX_DOCUMENT_LENGTH]
            builder.add_document_module(f"{_document_header}{i}", document_str)

        # Set assistant reply
        builder.set_assistant_description(_document_assistant_description)

        # Generate the XML string
        xml_string = builder.generate_xml()
        # Write the XML string to a file
        # - check the file path and make sure it is valid
        if not os.path.exists(os.path.dirname(xml_path)):
            os.makedirs(os.path.dirname(xml_path))
        if os.path.exists(xml_path):
            os.remove(xml_path)
        with open(f"{xml_path}", "w") as f:
            f.write(xml_string)
        pass

    def get_documents(self, xml_path=os.path.join(document_summary_path, "./schema_summary_sample.xml")):
        """
        Return a list of paths for XML files that need to be cached.
        """
        return [xml_path]

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