import gc

from typing import List, Tuple, Union, Dict, Optional
from tqdm import tqdm
import itertools

import torch

from .model import LanguageModel
from .prompt import Prompt, ModuleRef
from .schema import Parameter, TokenSequence, UnionModule, Schema, Path, Module

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


def pad_batch(batch_list: List[List[int]], pad_id: int) -> Tuple[List[List[int]], List[List[int]]]:
    max_len = max(map(len, batch_list))

    padded_batch = []
    mask_batch = []
    for batch in batch_list:
        padded_batch.append(batch + [pad_id] * (max_len - len(batch)))
        mask_batch.append([1] * len(batch) + [0] * (max_len - len(batch)))

    return padded_batch, mask_batch


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
    def __init__(self, max_ctx_length: int, num_layers: int, num_head: int, head_dim: int, target_device: torch.device):

        self.max_ctx_length = max_ctx_length
        self.num_head = num_head
        self.head_dim = head_dim

        self.device_cache = [
            (torch.empty(num_head, max_ctx_length, head_dim, device=target_device, dtype=torch.half),  # key
             torch.empty(num_head, max_ctx_length, head_dim, device=target_device, dtype=torch.half)) for _ in
            range(num_layers)]

        # print(num_head, max_ctx_length, head_dim)

        # stores staged modules
        self.staged = []
        self.length = 0

    @torch.inference_mode()
    def update(self, modules: List[TokenSequenceCache]):

        # TODO: adopt in-place sorting to reduce redundant host-device memory copies

        # cache rearrangement -> becomes new layout
        modules_ordered = sorted(modules, key=lambda e: e.usage_counter, reverse=True)

        retained = []

        for (m, m_prev) in zip(modules_ordered, self.staged):
            if m.token_sequence == m_prev.token_sequence:
                retained.append(m)
            else:
                break

        offset = sum(map(len, retained))
        updates = modules_ordered[len(retained):]

        # update the cache
        for m in updates:
            st = offset
            ed = st + len(m)

            for i in range(len(self.device_cache)):
                k_cache_tgt, v_cache_tgt = self.device_cache[i]
                k_cache_src, v_cache_src = m.cache[i]

                # print('k_src', k_cache_src.shape)
                # print('v_src', v_cache_src.shape)
                # print('k_tgt', k_cache_tgt.shape)
                # print('v_tgt', v_cache_tgt.shape)

                k_cache_tgt[:, st:ed, :].copy_(k_cache_src, non_blocking=True)
                v_cache_tgt[:, st:ed, :].copy_(v_cache_src, non_blocking=True)

            offset += len(m)

        # re-organize the cache

        self.staged = modules
        self.length = offset

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

    lm: LanguageModel

    def __init__(self, schema: Schema, lm: LanguageModel, batch_size: int = 1, target_device=None, no_cache=False):
        self.schema = schema
        self.lm = lm
        self.cache_l1 = dict()
        self.cache_l2 = dict()
        self.target_device = lm.device if target_device is None else target_device

        if not no_cache:
            self._process(batch_size)

    @torch.inference_mode()
    def _process(self, batch_size: int = 1):

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

        batch_path = []
        batch_token_ids = []
        batch_position_ids = []
        for k in range(len(paths_l1)):

            path = paths_l1[k]

            scaffold = self.schema.get_scaffold(path)

            token_ids = scaffold.token_ids()
            position_ids = scaffold.position_ids()

            # add to batch
            batch_path.append((path, scaffold))
            batch_token_ids.append(token_ids)
            batch_position_ids.append(position_ids)

            # batch bucket filled or the last iteration
            if len(batch_token_ids) == batch_size or k == len(paths_l1) - 1:

                # position_ids, token_ids = pad_unk(position_ids, token_ids, self.schema.tokenizer.hf_tokenizer.eos_token_id)
                # print(token_ids)

                # replace modeling_llama.py line 334
                #         cos, sin = self.rotary_emb(value_states, seq_len=torch.max(position_ids) + 1)

                batch_token_ids_padded, attn_mask = pad_batch(batch_token_ids, self.lm.eos_token_id)
                batch_position_ids_padded, _ = pad_batch(batch_position_ids, 0)

                d_output = self.lm(
                    input_ids=torch.tensor(batch_token_ids_padded, device=self.lm.device, dtype=torch.long),
                    position_ids=torch.tensor(batch_position_ids_padded, device=self.lm.device, dtype=torch.long),
                    attention_mask=torch.tensor(attn_mask, device=self.lm.device, dtype=torch.float16),
                    use_cache=True
                )

                # print(d_output.past_key_values[0].shape)
                # print(d_output.past_key_values[1].shape)

                kv_cache = d_output.past_key_values

                # print('num_layers', len(kv_cache))
                # print('k_shape', kv_cache[0][0].shape)
                # print('v_shape', kv_cache[0][1].shape)

                # iterate through all leaf nodes in target scaffold

                for j in range(len(batch_path)):

                    path, scaffold = batch_path[j]
                    position_ids = batch_position_ids[j]

                    #print(f"Caching module @{self.schema.name}/{path} ({len(position_ids)} tokens)...")

                    target = scaffold.select(path)

                    for tc in target.all_token_sequences():

                        offset = tc.offset
                        length = len(tc)

                        # why not just use tc.offset?
                        # this is because the offset is not always the same as the position_ids
                        # they might be mixed up. (but each token sequence is guaranteed to be continuous)
                        st = position_ids.index(offset)
                        ed = st + length

                        tc_cache = []

                        for k in range(len(kv_cache)):
                            k_cache = self.lm.store_k_hook(kv_cache[k][0])
                            v_cache = self.lm.store_v_hook(kv_cache[k][1])

                            k_cache_tc = k_cache[j, :, st:ed, :].squeeze(0).detach()
                            v_cache_tc = v_cache[j, :, st:ed, :].squeeze(0).detach()

                            if self.target_device != 'cpu':
                                k_cache_tc = k_cache_tc.cpu()
                                v_cache_tc = v_cache_tc.cpu()

                            tc_cache.append((k_cache_tc, v_cache_tc))

                        self.cache_l1[id(tc)] = TokenSequenceCache(tc, tc_cache)

                if self.target_device != 'cpu':
                    del d_output

                # clear batch
                batch_path = []
                batch_token_ids = []
                batch_position_ids = []

        gc.collect()
        torch.cuda.empty_cache()
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
    lm: LanguageModel
    schemas: Dict[str, SchemaCache]

    prompt_cache: PromptCache

    def __init__(self, max_ctx_length: int, lm: LanguageModel, target_device=None):

        self.lm = lm
        self.schemas = dict()
        self.target_device = lm.device if target_device is None else target_device

        num_layers, num_head, head_dim = lm.get_cache_shape()

        self.prompt_cache = PromptCache(
            max_ctx_length=max_ctx_length,
            num_layers=num_layers,
            num_head=num_head,
            head_dim=head_dim,
            target_device=self.target_device
        )

    def add_schema(self, schema: Union[str, Schema],
                   batch_size: int = 1,
                   max_tokens: Optional[int] = None,
                   no_cache: bool = False):

        if type(schema) == str:
            schema = Schema(schema, self.lm, max_tokens=max_tokens)

        if schema.name in self.schemas:
            raise ValueError(f'There is already a schema named {schema.name} in the cache')

        self.schemas[schema.name] = SchemaCache(schema, self.lm, batch_size, target_device=self.target_device,
                                                no_cache=no_cache)

    def get_schema(self, name: str) -> Optional[Schema]:
        if name not in self.schemas:
            return None
        return self.schemas[name].schema

    def remove_schema(self, name: str):
        if name not in self.schemas:
            raise ValueError(f'There is no such schema named {name} in the cache')

        del self.schemas[name]
        gc.collect()
        torch.cuda.empty_cache()

    def remove_all_schemas(self):

        # remove all schemas
        self.schemas = dict()

        gc.collect()
        torch.cuda.empty_cache()

    def process(self, prompt: Prompt, no_cache: bool = False, return_full_position_ids: bool = False) -> Tuple[
        List[int], List[int], float, Optional[KVCache]]:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        # assert that root tag matches engine signature
        if prompt.schema not in self.schemas:
            raise ValueError(f'There is no such layout named {prompt.schema} in the cache')

        cached = self.schemas[prompt.schema]
        schema = cached.schema

        orig_ids_list = []
        orig_pos_ids_list = []

        used_sequences = []
        argument_ids_list = []
        argument_pos_ids_list = []

        # first add root level modules
        stack: List[(ModuleRef, Module)] = [(prompt, schema)]

        while len(stack) > 0:
            ref, module = stack.pop()

            # step 1. first add leaf nodes
            for m in module.token_sequences():
                # kv_cache_list.append(cached.get_cache_l1(m))
                used_sequences.append(m)

                if no_cache or return_full_position_ids:
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

                argument_ids = self.lm.encode(arg.value)

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

        # print(prompt.text)

        # aa = self.lm.hf_tokenizer.tokenize(prompt.text)
        # print(aa)
        # aa = self.lm.hf_tokenizer.tokenize('\n')
        # print('newline,', aa)

        if len(prompt.text) > 0:
            text_token_ids = self.lm.encode(prompt.text)
            text_position_ids = list(range(len(schema), len(schema) + len(text_token_ids)))

            argument_ids_list.append(text_token_ids)
            argument_pos_ids_list.append(text_position_ids)

        input_ids = list(itertools.chain(*argument_ids_list))
        position_ids = list(itertools.chain(*argument_pos_ids_list))

        if no_cache:
            orig_input_ids = list(itertools.chain(*orig_ids_list))
            orig_position_ids = list(itertools.chain(*orig_pos_ids_list))

            sorted_pairs = sorted(zip(orig_position_ids + position_ids, orig_input_ids + input_ids))

            # Unpack the sorted pairs into two lists
            orig_position_ids, orig_input_ids = zip(*sorted_pairs)

            end.record()
            torch.cuda.synchronize()
            cache_time = start.elapsed_time(end)

            # print(f'Cache overhead: {cache_time:.2f} ms')

            vv = list(range(len(orig_position_ids)))

            return orig_input_ids, vv, cache_time, None
        else:

            used_seq_caches = []

            for s in used_sequences:
                seq_cache = cached.get_cache_l1(s)

                seq_cache.inc_usage_counter()
                used_seq_caches.append(seq_cache)

            # update prompt cache. this incurs some memcpy overhead.
            self.prompt_cache.update(used_seq_caches)
            cache = self.prompt_cache.cache
            end.record()
            torch.cuda.synchronize()
            cache_time = start.elapsed_time(end)

            # apply read hook
            for i in range(len(cache)):
                cache[i] = (self.lm.read_k_hook(cache[i][0]), self.lm.read_v_hook(cache[i][1]))

            # print(f'Cache overhead: {cache_time:.2f} ms')

            if return_full_position_ids:
                orig_position_ids = list(itertools.chain(*orig_pos_ids_list))
                position_ids = orig_position_ids + position_ids

            # print(orig_position_ids)
            return input_ids, position_ids, cache_time, cache
