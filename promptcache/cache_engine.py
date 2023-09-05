import lxml
import lxml.etree
from typing import List, Tuple, Union, Dict, cast
from tqdm import tqdm

import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM
)

from .prompt import Preprocessor, Prompt, ModuleRef
from .schema import Parameter, Module, UnionModule, Schema, Tokenizer, TokenSequence, Path

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CachedSchema:
    schema: Schema
    cache_l1: Dict[int, KVCache]
    cache_l2: Dict[Tuple[int, int], Tuple[KVCache, KVCache]]

    model: LlamaForCausalLM

    def __init__(self, schema: Schema, model: LlamaForCausalLM):
        self.schema = schema
        self.model = model
        self.cache_l1 = dict()
        self.cache_l2 = dict()

        self._process()

    def _process(self):

        # Get all possible L1 scaffolds
        stack = list()
        paths_l1 = [Path(), ]

        if self.schema.contains_union():
            stack.append((list(), True, self.schema))

        while len(stack) > 0:
            path, is_default_parent, u = stack.pop()

            for e in u.children:
                if type(e) == Module and e.contains_union():
                    stack.append((path + [u.name], is_default_parent, e))

                elif type(e) == UnionModule:
                    for n in e.modules:
                        is_default = e.scaffold_name == n.name and is_default_parent

                        if n.contains_union():
                            stack.append((path + [u.name], is_default, n))

                        if not is_default:
                            paths_l1.append(Path(path + [u.name, n.name]).next)

        # For each path, update every leaf nodes (token sequence) under that path
        for path in paths_l1:

            print("Processing..", path)
            scaffold = self.schema.get_scaffold(path)

            token_ids = scaffold.token_ids()
            position_ids = scaffold.position_ids()

            # xxx = torch.tensor([token_ids], device=self.model.device, dtype=torch.long)
            # vvv = self.model.get_input_embeddings()(xxx)
            # print(vvv.shape)
            # print('no issue')

            d_output = self.model(
                input_ids=torch.tensor([token_ids], device=self.model.device, dtype=torch.long),
                # position_ids=torch.tensor([position_ids], device=self.model.device, dtype=torch.long),
            )

            # print(d_output.past_key_values[0].shape)
            # print(d_output.past_key_values[1].shape)

            k_cache, v_cache = d_output.past_key_values[0]

            # iterate through all leaf nodes in target scaffold
            target = scaffold.select(path)

            for tc in target.all_token_sequences():
                offset = tc.offset
                length = len(tc)

                self.cache_l1[id(tc)] = k_cache[offset:offset + length], v_cache[offset:offset + length]

    def get_cache_l1(self, seq: TokenSequence) -> Union[KVCache, None]:
        seq_id = id(seq)
        if seq_id not in self.cache_l1:
            return None
        return self.cache_l1[seq_id]

    def get_cache_l2(self, seq1: TokenSequence, seq2: TokenSequence) -> Union[Tuple[KVCache, KVCache], None]:
        seq1_id, seq2_id = max(id(seq1), id(seq2)), min(id(seq1), id(seq2))
        if (seq1_id, seq2_id) not in self.cache_l2:
            return None
        return self.cache_l2[(seq1_id, seq2_id)]


class CacheEngine:
    model: LlamaForCausalLM
    tokenizer: Tokenizer
    schemas: Dict[str, CachedSchema]

    def __init__(self, model: LlamaForCausalLM, tokenizer: Tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.schemas = dict()

    def add_schema(self, schema: Union[str, Schema]):
        if type(schema) == str:
            schema = Schema(schema, self.tokenizer)

        if schema.name in self.schemas:
            raise ValueError(f'There is already a schema named {schema.name} in the cache')

        self.schemas[schema.name] = CachedSchema(schema, self.model)

    def get_schema(self, name: str) -> Union[Schema, None]:
        if name not in self.schemas:
            return None
        return self.schemas[name].schema

    def process(self, prompt: Prompt) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:

        # assert that root tag matches engine signature
        if prompt.schema not in self.schemas:
            raise ValueError(f'There is no such layout named {prompt.schema} in the cache')

        cached = self.schemas[prompt.schema]
        schema = cached.schema

        kv_cache_list = []
        argument_ids_list = []
        argument_pos_ids_list = []

        # first add root level modules
        stack: List[(ModuleRef, Module)] = [(prompt, schema)]

        for m in prompt.modules:

            module = schema.select(m.name)
            if module is None:
                raise ValueError(f'There is no such module named {m.name} in the schema {schema.name}')

            stack.append((m, module))

        while len(stack) > 0:
            ref, module = stack.pop()

            # step 1. first add leaf nodes
            for m in module.token_sequences():
                kv_cache_list.append(cached.get_cache_l1(m))

            # step 2. process parameter-argument pairs
            parameters = module.parameters()
            for arg in ref.args:

                parameter = None
                for p in parameters:
                    if p.name == arg.name:
                        parameter = p
                        break

                if parameter is None:
                    raise ValueError(f'There is no such parameter named {arg.name} in the module {module.name}')

                argument_ids = self.tokenizer.encode(arg.value)

                if len(argument_ids) > parameter.length:
                    raise ValueError(
                        f'The argument {arg.name} is too long. It should be at most {parameter.length} characters long')

                argument_pos_ids = parameter.position_ids()[:len(argument_ids)]

                argument_ids_list.append(argument_ids)
                argument_pos_ids_list.append(argument_pos_ids)

            # step 3. update stack
            for m in ref.modules:
                module = schema.select(m.name)
                if module is None:
                    raise ValueError(f'There is no such module named {m.name} in the schema {schema.name}')

                stack.append((m, module))

        # add trailing text
        text_token_ids = self.tokenizer.encode(prompt.text)
        text_position_ids = torch.arange(len(schema), len(schema) + len(text_token_ids))

        argument_ids_list.append(text_token_ids)
        argument_pos_ids_list.append(text_position_ids)

        input_ids = torch.tensor(argument_ids_list, dtype=torch.long).view(-1)
        position_ids = torch.tensor(argument_pos_ids_list, dtype=torch.long).view(-1)

        k_cache = torch.cat([kv_cache[0] for kv_cache in kv_cache_list])
        v_cache = torch.cat([kv_cache[1] for kv_cache in kv_cache_list])

        kv_cache = (k_cache, v_cache)

        return input_ids, position_ids, kv_cache
