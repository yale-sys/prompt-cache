# import re
#
# from .benchmark_base import Benchmark, Entry
# from .utils import XMLSchemaBuilder
# from datasets import load_dataset
# import os
#
#
# def escape_tags(input_str):
#     # pattern = r'<(?P<content>.*?)>'
#
#     # # The lambda function ensures only the first letter is capitalized
#     # def repl(match):
#     #     return '(' + match.group("content").capitalize() + ')'
#     #
#     # return re.sub(pattern, repl, input_str)
#     return input_str.replace('<', '(').replace('>', ')')
#
#
# class LongBench(Benchmark):
#     def __init__(self, subset_name: str):
#         super().__init__(subset_name)
#
#     def init(self, limit_entries=None):
#         """
#         Download (one time) and load the dataset to run;
#         Preprocess the dataset to be organized in the `Entry` format.
#         """
#         self.dataset = load_dataset('THUDM/LongBench', self.dataset_name)
#
#         count = 0
#         for split in self.dataset.values():
#             for item in split:
#                 if limit_entries is not None and count >= limit_entries:
#                     break
#                 schema_name = f"schema_{item['_id']}"
#
#                 fmt_schema = self.dataset_prompt["context"].format(context=escape_tags(item["context"]))
#                 fmt_prompt = self.dataset_prompt["question"].format(input=escape_tags(item["input"])[:1000])
#                 fmt_question = self.dataset_prompt["answer"]
#
#                 schema = f"""
# <schema name="{schema_name}"><system/><user><module name="context">{fmt_schema}</module></schema>
# """
#
#                 prompt = f"""
# <prompt schema='{schema_name}'><context/>{fmt_prompt}</user><assistant>{fmt_question}</prompt>
# """
#                 self.entries.append(Entry(schema, prompt, item["answers"]))
#
#                 count += 1

import re

from .benchmark_base import Benchmark, Entry
from .utils import XMLSchemaBuilder
from datasets import load_dataset
import os

_system_description = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, \
honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with \
almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false \
or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the \
assistant is practical and really does its best, and doesn't let caution get too much in the way of being \
useful."
_user_description = "For the upcoming interaction, I would like you to answer some questions about the document."
_assistant_description = "Sure. I have read the document. Please give me any question."


def escape_tags(input_str):
    # pattern = r'<(?P<content>.*?)>'

    # # The lambda function ensures only the first letter is capitalized
    # def repl(match):
    #     return '(' + match.group("content").capitalize() + ')'
    #
    # return re.sub(pattern, repl, input_str)
    return input_str.replace('<', '(').replace('>', ')')


class LongBench(Benchmark):
    def __init__(self, subset_name: str):
        super().__init__(subset_name)

    def init(self, limit_entries=None):
        """
        Download (one time) and load the dataset to run;
        Preprocess the dataset to be organized in the `Entry` format.
        """
        self.dataset = load_dataset('THUDM/LongBench', self.dataset_name)

        count = 0
        for split in self.dataset.values():
            for item in split:
                if limit_entries is not None and count >= limit_entries:
                    break
                id = item["_id"]
                schema_name = f"schema_{id}"

                fmt_schema = self.dataset_prompt["context"].format(context=escape_tags(item["context"]))

                schema_content = f"""
<schema name="{schema_name}">
<system/>
<user><module name="context">
{fmt_schema}
</module>
</schema>
"""
                #
                #
                # builder = XMLSchemaBuilder(schema_name)
                # context = item["context"]
                # # print(len(context))
                # # title = item["title"]
                # question = item["input"]
                # answer = item["answers"]
                # builder.set_system_description(_system_description)
                # builder.set_user_description(_user_description)
                # builder.add_document_module("context",
                #                             self.dataset_prompt["context"].format(context=escape_tags(context)))
                # builder.set_assistant_description(_assistant_description)

                schema_file_name = f"{schema_name}.xml"
                with open(os.path.join(self.schema_path, schema_file_name), "w") as f:
                    f.write(schema_content)

                prompt = f"""
                <prompt schema='{schema_name}'>
                <context/>
                {self.dataset_prompt["question"].format(input=escape_tags(item["input"])[:1000])}</user><assistant>{self.dataset_prompt["answer"]}</prompt>
                """
                self.entries.append(Entry(schema_file_name, prompt, item["answers"]))

                count += 1


if __name__ == '__main__':
    sq = LongBench('narrativeqa')
    sq.init()
    print(sq.get_entry_count())
    print(sq.get_query((100, 101)))
