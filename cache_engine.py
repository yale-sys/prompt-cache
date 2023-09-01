import lxml
import lxml.etree
from typing import List, Tuple, Union, Dict, cast
from tqdm import tqdm

import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM
)

from prompt import Preprocessor, Prompt
from schema import Parameter, Module, UnionModule, Schema, Tokenizer, TokenSequence, Path

ElementId = int
KVCache = Tuple[torch.Tensor, torch.Tensor]


class CachedSchema:
    schema: Schema
    cache_l1: Dict[ElementId, KVCache]
    cache_l2: Dict[(ElementId, ElementId), (KVCache, KVCache)]

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
                    stack.append((path + [e.name], is_default_parent, e))

                elif type(e) == UnionModule:
                    for n in e.modules:
                        is_default = e.scaffold_name == n.name and is_default_parent

                        if n.contains_union():
                            stack.append((path + [e.name], is_default, n))

                        if not is_default:
                            paths_l1.append(path + [e.name])

        # For each path, update every leaf nodes (token sequence) under that path
        for path in paths_l1:

            scaffold = self.schema.get_scaffold(path)

            token_ids = scaffold.token_ids()
            position_ids = scaffold.position_ids()

            d_output = self.model(
                input_ids=torch.LongTensor([token_ids], device=self.model.device),
                position_ids=torch.LongTensor([position_ids], device=self.model.device),
            )

            k_cache, v_cache = d_output.past_key_values

            # iterate through all leaf nodes in target scaffold
            target = scaffold.select(path)

            for tc in target.token_sequences():
                offset = tc.offset
                length = len(tc)

                self.cache_l1[id(tc)] = k_cache[offset:offset + length], v_cache[offset:offset + length]

    def get_cache_l1(self, seq: TokenSequence) -> Union[KVCache, None]:
        seq_id = id(seq)
        if seq_id not in self.cache_l1:
            return None
        return self.cache_l1[seq_id]

    def get_cache_l2(self, seq1: TokenSequence, seq2: TokenSequence) -> Union[(KVCache, KVCache), None]:
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

        # first add anonymous modules
        for m in schema.modules:
            if type(m) == Module:
                if m.is_anonymous:
                    kv_cache = cached.get_cache(m.name)
                    kv_cache_list.append(kv_cache)

        # iterate through each tags in root
        used_modules = []

        for el in root:

            module = schema.find_module(el.tag)

            if module is None:
                raise ValueError(f'There is no such module named {el.tag} in the layout {root.tag}')

            offset = cached.get_offset(module.name)
            kv_cache = cached.get_cache(module.name)
            kv_cache_list.append(kv_cache)

            # check union validity
            if offset in [cached.get_offset(m.name) for m in used_modules]:
                raise ValueError(
                    f"Only one module from union can be used: {module.name} cannot be used with {m.name}")

            used_modules.append(module)

            # check parameters
            for attr_name, attr_value in el.attrib.items():

                parameter = module.find_parameter(attr_name)

                if parameter is None:
                    raise ValueError(f'There is no such parameter named {attr_name} in the module {module.name}')

                # check length
                argument_ids = self.tokenizer.encode(attr_value, add_special_tokens=False, return_tensors='pt')[0]

                if len(argument_ids) > parameter.length:
                    raise ValueError(
                        f'The argument {attr_name} is too long. It should be at most {parameter.length} characters long')

                argument_pos_ids = parameter._position_ids[:len(argument_ids)] + offset

                argument_ids_list.append(argument_ids)
                argument_pos_ids_list.append(argument_pos_ids)

        tail_offset = cached.offset + schema.length
        tail_input_ids = self.tokenizer.encode(root.tail, add_special_tokens=False, return_tensors='pt')[0]
        tail_position_ids = torch.arange(tail_offset, tail_offset + len(tail_input_ids))

        argument_ids_list.append(tail_input_ids)
        argument_pos_ids_list.append(tail_position_ids)

        input_ids = torch.cat(argument_ids_list)
        position_ids = torch.cat(argument_pos_ids_list)
        k_cache = torch.cat([kv_cache[0] for kv_cache in kv_cache_list])
        v_cache = torch.cat([kv_cache[1] for kv_cache in kv_cache_list])

        kv_cache = (k_cache, v_cache)

        return input_ids, position_ids, kv_cache
