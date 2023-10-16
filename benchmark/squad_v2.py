from .benchmark_base import Benchmark, Entry
from .utils import XMLSchemaBuilder
from datasets import load_dataset
import os

_document_schema_name = "document_summary"
_document_header = "Document"
_document_system_description = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, \
honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with \
almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false \
or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the \
assistant is practical and really does its best, and doesn't let caution get too much in the way of being \
useful."
_document_user_description = "For the upcoming interaction, I would like you to answer some questions about the document."
_document_assistant_description = "Sure. I have read the document. Please give me any question."


class SquadV2(Benchmark):
    def __init__(self):
        super().__init__("squad_v2")

    def init(self, limit_entries=None):
        """
        Download (one time) and load the dataset to run; 
        Preprocess the dataset to be organized in the `Entry` format.
        """
        self.dataset = load_dataset(self.dataset_name)
        count = 0
        # for split in self.dataset.values():
        # only use validation set
        for item in self.dataset['validation']:
            if limit_entries is not None and count >= limit_entries:
                break
            id = item["id"]
            schema_name = f"schema_{id}"
            builder = XMLSchemaBuilder(schema_name)
            context = item["context"]
            # title = item["title"]
            question = item["question"]
            answer = item["answers"]["text"]
            builder.set_system_description(_document_system_description)
            builder.set_user_description(_document_user_description)
            builder.add_document_module("context", self.dataset_prompt["context"].format(context=context))
            builder.set_assistant_description(_document_assistant_description)

            schema_file_name = f"{schema_name}.xml"
            with open(os.path.join(self.schema_path, schema_file_name), "w") as f:
                f.write(builder.generate_xml())

            # skip entry without ground truth
            if len(answer) == 0:
                continue

            prompt = f"""
            <prompt schema='{schema_name}'>
            <context/>
            <user>{self.dataset_prompt["question"].format(input=question)}</user></prompt>
            """
            self.entries.append(Entry(schema_file_name, prompt, answer))

            count += 1

if __name__ == '__main__':
    sq = SquadV2()
    sq.init()
    print(sq.get_entry_count())
