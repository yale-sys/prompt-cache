# Add the parent directory to the sys.path list
import os, sys
document_summary_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(document_summary_path, '..')))

import unittest
from benchmark.profile_parser import JsonParser, BenchmarkProfileParser
class TestJsonParser(unittest.TestCase):
    def setUp(self):
        self.parser = JsonParser('./benchmark/profiles/document_summary_simple.json')
        self.parser.parse()

    def test_data_type(self):
        self.assertIsInstance(self.parser.get_data(), dict)

    def test_data_content(self):
        expected_keys = ['benchmark_name', 'benchmark_description', 'benchmark_dataset_name',
                         'benchmark_dataset_comment', 'dataset_size', 'prompt_cache']
        self.assertListEqual(list(self.parser.get_data().keys()), expected_keys)

class TestBenchmarkProfileParser(unittest.TestCase):
    def setUp(self):
        self.parser = BenchmarkProfileParser('./benchmark/profiles/document_summary_simple.json')
        self.parser.parse()

    def test_benchmark_name(self):
        self.assertEqual(self.parser.get_benchmark_name(), 'Document summary simple')


from promptcache.model import Llama2, Falcon, Mpt
from transformers import (
    AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
)
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template

profile_path = './benchmark/profiles/document_summary_simple.json'

def run_sample_test(disable_prompt_cache):
    ### Configurations ###
    disable_cuda = False

    ######################
    lm = Llama2("meta-llama/Llama-2-7b-chat-hf",
                load_in_8bit=True if not disable_cuda else False,
                device_map="auto" if not disable_cuda else None)

    cache_engine = CacheEngine(2500, lm)
    gen_engine = GenerationEngine(lm)

    preproc = [
        CompactSpaces(),
        lm.get_formatter()
    ]

    # Parameter setup
    parameter = GenerationParameters(
        temperature=0.1,
        repetition_penalty=1.17,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    # Schema and prompt text setup
    # instatiate and init
    # dataset_loader.get_cacheable_xml()
    print("prompt cache: ", not disable_prompt_cache)
    cache_engine.add_schema(read_file("./benchmark/document_summary/schema_summary_sample.xml", preproc))
    # prompt_text = "<prompt schema='document_summary'> <Document0/>"
    
    for document_idx in range(10):
        prompt_text = f'''
        <prompt schema='document_summary'>
        <Document{document_idx}/>
        <user>Summarize the above document in around THREE sentences:</user>
        </prompt>
        '''

        # user portion of the prompt
        # prompt_text += f'<user> Provide only summary and do not ask back </user>'
        # print(prompt_text)
        prompt = Prompt(prompt_text, preproc)
        # print(prompt)
        token_ids, position_ids, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache,
                                                            return_full_position_ids=lm.use_full_position_ids)
        if disable_prompt_cache:
            assert cache is None

        output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2,
                                            use_full_position_ids=lm.use_full_position_ids)

        print(f"Assistant: ", end="", flush=True)

        resp = ""
        pre = 0
        for outputs in output_stream:
            output_text = outputs.new_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                tt = " ".join(output_text[pre:now])
                resp += tt + " "
                print(tt, end=" ", flush=True)
                pre = now

        print("\n")
    pass

from benchmark.document_summary import DocumentSummary

class TestDocumentSummary(unittest.TestCase):
    def setUp(self):
        self.parser = BenchmarkProfileParser(profile_path)
        self.parser.parse()
        self.document_summary = DocumentSummary()
        self.document_summary.init(verbose=False)

    def test_load_config(self):
        self.assertIsInstance(self.parser.get_data(), dict)

    def test_llm_with_dataset(self):
        run_sample_test(not self.parser.get_data()['prompt_cache'])

if __name__ == '__main__':
    unittest.main()