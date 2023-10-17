import re

from .benchmark_base import Benchmark, Entry
from .utils import XMLSchemaBuilder
from datasets import load_dataset
import os


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
                schema_name = f"schema_{item['_id']}"

                fmt_schema = self.dataset_prompt["context"].format(context=escape_tags(item["context"]))
                fmt_prompt = self.dataset_prompt["question"].format(input=escape_tags(item["input"])[:1000])
                fmt_question = self.dataset_prompt["answer"]

                schema = f"""
<schema name="{schema_name}"><system/><user><module name="context">{fmt_schema}</module></schema>
"""

                prompt = f"""
<prompt schema='{schema_name}'><context/>{fmt_prompt}</user><assistant>{fmt_question}</prompt>
"""
                self.entries.append(Entry(schema, prompt, item["answers"]))

                count += 1
