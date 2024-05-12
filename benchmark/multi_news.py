import sys
import os

# Add the parent directory to the sys.path list
document_summary_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.abspath(os.path.join(document_summary_path, '..')))

from .benchmark_base import Benchmark, Entry
from .dataset_download import load_documentation_summary
from .utils import XMLSchemaBuilder

_document_header = "Document"
_document_dataset = "validation"
_document_system_description = "Dialogues between a user and an AI about the document provided by the user with the aim of being helpful, aware, and accurate."
_document_assistant_description = "Sure. I have read the document. give me any instructions regarding summarization, and I will try to follow them."
_document_user_summary = "Summarize the above document in around THREE sentences:"

MAX_DOCUMENT_LENGTH = 2560

class MultiNews(Benchmark):
    def __init__(self):
        super().__init__("multi_news")
        self.next_query_idx = 0

    def init(self, limit_entries=None, verbose=False):
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
        schema_file_name = "schema_summary_sample.xml"
        self._generate_xml(limit_entries)

    def _generate_xml(self, limit_entries):
        # Generate xml files
        # - In this version, we build the xml file per entry
        count = 0
        for document_idx in range(len(self.dataset[_document_dataset])):
            if limit_entries is not None and count >= limit_entries:
                break
            # Create an instance of XMLSchemaBuilder with the schema name "document_summary"
            schema_name = f"_document_schema_name_{document_idx}"
            builder = XMLSchemaBuilder(schema_name)

            # Set the system description
            builder.set_system_description(_document_system_description)

            # Set the user description
            builder.set_user_description("")  # _document_user_description

            # Add document modules
            # builder.add_document_module("DOC", "The given documents are the target for the summarization task. They can contain multiple sentences.")
            document_str = self.dataset[_document_dataset][document_idx]["document"].replace("’", "'").replace("”",
                                                                                                               '"').replace(
                "“", '"').replace("‘", "'").replace("…", "...").replace("–", "-")
            if len(document_str) > MAX_DOCUMENT_LENGTH:
                document_str = document_str[:MAX_DOCUMENT_LENGTH]
            builder.add_document_module(f"{_document_header}{document_idx}", document_str)

            # Set assistant reply
            builder.set_assistant_description(_document_assistant_description)

            # Write the XML string to a file
            schema_file_name = f"{schema_name}.xml"
            with open(os.path.join(self.schema_path, schema_file_name), "w") as f:
                f.write(builder.generate_xml())

            # Prepare the entry        
            prompt = f"""
            <prompt schema='{schema_name}'>
            <{_document_header}{document_idx}/>
            <user>{_document_user_summary}</user></prompt>
            """
            summary_str = self.dataset[_document_dataset][document_idx]["summary"].replace("’", "'").replace("”",
                                                                                                             '"').replace(
                "“", '"').replace("‘", "'").replace("…", "...").replace("–", "-")
            self.entries.append(Entry(schema_file_name, prompt, summary_str))

            count += 1

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
        raise NotImplementedError(
            "This method should call utility function to measure how the response is closer to the expected answer.")

if __name__ == '__main__':
    mn = MultiNews()
    mn.init()
    print(mn.get_entry_count())
