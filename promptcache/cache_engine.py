import lxml
import lxml.etree
from typing import List, Tuple, Union, Dict, cast, Optional
from tqdm import tqdm
import itertools

import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM
)

from .prompt import Preprocessor, Prompt, ModuleRef
from .schema import Parameter, Module, UnionModule, Schema, Tokenizer, TokenSequence, Path

# list - each decoding layer in transformer
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def pad_unk(position_ids: List[int], token_ids: List[int], unk_token_id: int):
    print('before')
    print(position_ids)
    print(token_ids)

    pos_min = min(position_ids)
    pos_max = max(position_ids)

    padded_position_ids = list(range(pos_min, pos_max + 1))
    padded_token_ids = [unk_token_id] * len(padded_position_ids)

    for i, token_id in zip(position_ids, token_ids):
        padded_token_ids[padded_position_ids.index(i)] = token_id

    print('after')
    print(padded_position_ids)
    print(padded_token_ids)

    return padded_position_ids, padded_token_ids


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

    @torch.inference_mode()
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

            scaffold = self.schema.get_scaffold(path)

            token_ids = scaffold.token_ids()
            position_ids = scaffold.position_ids()

            # position_ids, token_ids = pad_unk(position_ids, token_ids, self.schema.tokenizer.hf_tokenizer.eos_token_id)

            # print(token_ids)

            print(f"Caching module @{path} ({len(token_ids)} tokens)...")

            # replace modeling_llama.py line 334
            #         cos, sin = self.rotary_emb(value_states, seq_len=torch.max(position_ids) + 1)

            d_output = self.model(
                input_ids=torch.tensor([token_ids], device=self.model.device, dtype=torch.long),
                position_ids=torch.tensor([position_ids], device=self.model.device, dtype=torch.long),
            )

            # print(d_output.past_key_values[0].shape)
            # print(d_output.past_key_values[1].shape)

            kv_cache = d_output.past_key_values
            # iterate through all leaf nodes in target scaffold
            target = scaffold.select(path)

            for tc in target.all_token_sequences():
                offset = tc.offset
                length = len(tc)

                # why not just use tc.offset?
                # this is because the offset is not always the same as the position_ids
                # they might be mixed up. (but each token sequence is guaranteed to be continuous)
                st = position_ids.index(offset)
                ed = st + length

                self.cache_l1[id(tc)] = [(kv_cache[i][0][:, :, st:ed, :].detach().cpu(),
                                          kv_cache[i][1][:, :, st:ed, :].detach().cpu())
                                         for i in range(len(kv_cache))]

    def get_cache_l1(self, seq: TokenSequence) -> Optional[KVCache]:
        seq_id = id(seq)
        if seq_id not in self.cache_l1:
            return None
        return self.cache_l1[seq_id]

    def get_cache_l2(self, seq1: TokenSequence, seq2: TokenSequence) -> Optional[Tuple[KVCache, KVCache]]:
        seq1_id, seq2_id = max(id(seq1), id(seq2)), min(id(seq1), id(seq2))
        if (seq1_id, seq2_id) not in self.cache_l2:
            return None
        return self.cache_l2[(seq1_id, seq2_id)]


class CacheEngine:
    model: LlamaForCausalLM
    tokenizer: Tokenizer
    schemas: Dict[str, CachedSchema]

    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizer):

        self.model = model
        self.tokenizer = Tokenizer(tokenizer)
        self.schemas = dict()

    def add_schema(self, schema: Union[str, Schema]):
        if type(schema) == str:
            schema = Schema(schema, self.tokenizer)

        if schema.name in self.schemas:
            raise ValueError(f'There is already a schema named {schema.name} in the cache')

        self.schemas[schema.name] = CachedSchema(schema, self.model)

    def get_schema(self, name: str) -> Optional[Schema]:
        if name not in self.schemas:
            return None
        return self.schemas[name].schema

    def process(self, prompt: Prompt) -> Tuple[List[int], List[int], KVCache, List[int], List[int]]:

        # assert that root tag matches engine signature
        if prompt.schema not in self.schemas:
            raise ValueError(f'There is no such layout named {prompt.schema} in the cache')

        cached = self.schemas[prompt.schema]
        schema = cached.schema

        orig_ids_list = []
        orig_pos_ids_list = []

        kv_cache_list = []
        argument_ids_list = []
        argument_pos_ids_list = []

        # first add root level modules
        stack: List[(ModuleRef, Module)] = [(prompt, schema)]

        # for m in prompt.modules:
        #
        #     module = schema.select(m.name)
        #     if module is None:
        #         raise ValueError(f'There is no such module named {m.name} in the schema {schema.name}')
        #
        #     stack.append((m, module))

        while len(stack) > 0:
            ref, module = stack.pop()

            # step 1. first add leaf nodes
            for m in module.token_sequences():
                kv_cache_list.append(cached.get_cache_l1(m))
                orig_ids_list.append(m.token_ids())
                orig_pos_ids_list.append(m.position_ids())

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

                argument_ids = self.tokenizer.encode_maxx(arg.value)

                if len(argument_ids) > parameter.length:
                    raise ValueError(
                        f'The argument {arg.name} is too long. It should be at most {parameter.length} characters long')

                argument_pos_ids = parameter.position_ids()[:len(argument_ids)]

                argument_ids_list.append(argument_ids)
                argument_pos_ids_list.append(argument_pos_ids)

            # step 3. update stack
            for m in ref.modules:
                submodule = module.select(m.name)
                if submodule is None:
                    raise ValueError(f'There is no such module named @{m.name} in the module @{module.name}')

                stack.append((m, submodule))

        # add trailing text
        text_token_ids = self.tokenizer.encode_maxx(prompt.text)
        text_position_ids = list(range(len(schema), len(schema) + len(text_token_ids)))

        argument_ids_list.append(text_token_ids)
        argument_pos_ids_list.append(text_position_ids)

        input_ids = list(itertools.chain(*argument_ids_list))
        position_ids = list(itertools.chain(*argument_pos_ids_list))

        orig_input_ids = list(itertools.chain(*orig_ids_list))
        orig_position_ids = list(itertools.chain(*orig_pos_ids_list))

        # print([kv_cache[0].shape for kv_cache in kv_cache_list])
        num_layers = len(kv_cache_list[0])

        out_kv_cache = []

        for i in range(num_layers):
            k_cache_i = torch.cat([kv_cache[i][0] for kv_cache in kv_cache_list], dim=2)
            v_cache_i = torch.cat([kv_cache[i][1] for kv_cache in kv_cache_list], dim=2)
            out_kv_cache.append((k_cache_i, v_cache_i))

        sorted_pairs = sorted(zip(orig_position_ids + position_ids, orig_input_ids + input_ids))

        # Unpack the sorted pairs into two lists
        orig_position_ids, orig_input_ids = zip(*sorted_pairs)

        # position_ids, token_ids = pad_unk(position_ids, input_ids, self.tokenizer.unk_token_id)

        return input_ids, position_ids, out_kv_cache, orig_input_ids, orig_position_ids
