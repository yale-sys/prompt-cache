import gc

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
from .schema import Parameter, TokenSequence, UnionModule, Schema, Tokenizer, Path

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


class TokenSequenceCache:
    token_sequence: TokenSequence
    host_cache: KVCache
    device_cache: Optional[KVCache] = None

    usage_counter: int = 0

    def __init__(self, seq: TokenSequence, cache: KVCache):
        self.token_sequence = seq
        self.host_cache = cache
        self.usage_counter = 0

    def inc_usage_counter(self):
        self.usage_counter += 1

    def upload(self, device: str):
        if self.device_cache is None:
            self.device_cache = [(kv[0].to(device, non_blocking=True),
                                  kv[1].to(device, non_blocking=True)) for kv in self.host_cache]

    def free(self):
        # need to invoke gc.collect() manually later
        if self.device_cache is not None:
            self.device_cache = None

    @property
    def cache(self) -> KVCache:
        # prioritize device cache
        if self.device_cache is None:
            return self.host_cache
        return self.device_cache

    def __len__(self):
        return len(self.token_sequence)


# this persists in the device memory
class PromptCache:
    staged: List[TokenSequenceCache]
    length: int

    # only device cache
    max_ctx_length: int
    num_head: int
    head_dim: int
    device_cache: KVCache

    # hidden_dim is usually num_head * head_dim
    def __init__(self, max_ctx_length: int, num_layers: int, num_head: int, head_dim: int, device: torch.device):

        self.max_ctx_length = max_ctx_length
        self.num_head = num_head
        self.head_dim = head_dim
        self.device_cache = [(torch.empty(num_head, max_ctx_length, head_dim, device=device),  # key
                              torch.empty(num_head, max_ctx_length, head_dim, device=device)) for _ in
                             range(num_layers)]

        # stores staged modules
        self.staged = []
        self.length = 0

    def update(self, modules: List[TokenSequenceCache]):

        # TODO: adopt in-place sorting to reduce redundant host-device memory copy

        # cache rearrangement -> becomes new layout
        modules_ordered = sorted(modules, key=lambda e: e.usage_counter, reverse=True)
        updates = []
        # staged modules are already sorted.

        # skip common park
        diff = False
        offset = 0
        for (m, m_prev) in zip(modules_ordered, self.staged):
            if m.token_sequence != m_prev.token_sequence:
                diff = True
            if diff:
                updates.append(m)
            else:
                offset += len(m)

        # update the cache
        for m in updates:
            st = offset
            ed = st + len(m)

            for i in range(len(self.device_cache)):
                self.device_cache[i][0][:, st:ed, :].copy_(m.device_cache[i][0], non_blocking=True)
                self.device_cache[i][1][:, st:ed, :].copy_(m.device_cache[i][1], non_blocking=True)

            offset += len(m)

        # re-organize the cache

        self.staged = modules
        self.length = offset

        # compare current setup with new setup
        # rewrite
        #    - from host
        #    - from device
        # use Tensor.copy_ to do this. (set non_blocking=True)

        # new cache blocks will be stored in the gpu.
        # then if the space runs out,
        #    - evict most unpopular ones.

    def __len__(self):
        return self.length

    @property
    def cache(self) -> KVCache:
        return [(self.device_cache[i][0][:, :self.length, :],
                 self.device_cache[i][1][:, :self.length, :])
                for i in range(len(self.device_cache))]


class SchemaCache:
    schema: Schema
    cache_l1: Dict[int, TokenSequenceCache]
    cache_l2: Dict[Tuple[int, int], Tuple[TokenSequenceCache, TokenSequenceCache]]

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
                if type(e) == TokenSequence and e.contains_union():
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

            print(f"Caching module @{self.schema.name}/{path} ({len(token_ids)} tokens)...")

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

                tc_cache = [(kv_cache[i][0][:, :, st:ed, :].squeeze(0).detach().cpu(),
                             kv_cache[i][1][:, :, st:ed, :].squeeze(0).detach().cpu())
                            for i in range(len(kv_cache))]

                self.cache_l1[id(tc)] = TokenSequenceCache(tc, tc_cache)

        # upload to gpu.

    def get_cache_l1(self, seq: TokenSequence) -> Optional[TokenSequenceCache]:
        seq_id = id(seq)
        if seq_id not in self.cache_l1:
            return None
        return self.cache_l1[seq_id]

    def get_cache_l2(self, seq1: TokenSequence, seq2: TokenSequence) -> Optional[
        Tuple[TokenSequenceCache, TokenSequenceCache]]:

        seq1_id, seq2_id = max(id(seq1), id(seq2)), min(id(seq1), id(seq2))
        if (seq1_id, seq2_id) not in self.cache_l2:
            return None
        return self.cache_l2[(seq1_id, seq2_id)]


# each cache block is either allocated in the host or device memory.
# only upload to GPU when needed..

# first all


class CacheEngine:
    model: LlamaForCausalLM
    tokenizer: Tokenizer
    schemas: Dict[str, SchemaCache]

    prompt_cache: PromptCache

    def __init__(self, max_ctx_length: int, model: LlamaForCausalLM, tokenizer: LlamaTokenizer):

        self.model = model
        self.tokenizer = Tokenizer(tokenizer)
        self.schemas = dict()

        self.prompt_cache = PromptCache(
            max_ctx_length=max_ctx_length,
            num_layers=model.config.num_hidden_layers,
            num_head=model.config.num_attention_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            device=model.device
        )

    def add_schema(self, schema: Union[str, Schema]):
        if type(schema) == str:
            schema = Schema(schema, self.tokenizer)

        if schema.name in self.schemas:
            raise ValueError(f'There is already a schema named {schema.name} in the cache')

        self.schemas[schema.name] = SchemaCache(schema, self.model)

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

        used_sequences = []
        kv_cache_list = []
        argument_ids_list = []
        argument_pos_ids_list = []

        # first add root level modules
        stack: List[(ModuleRef, TokenSequence)] = [(prompt, schema)]

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
                #kv_cache_list.append(cached.get_cache_l1(m))
                used_sequences.append(m)
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
