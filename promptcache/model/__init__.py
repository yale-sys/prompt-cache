import abc
from typing import Callable, List, Optional, Tuple
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, PreTrainedTokenizer, \
    PretrainedConfig, PreTrainedModel, CodeLlamaTokenizer

from promptcache.model.falcon import FalconForCausalLM
from promptcache.model.llama2 import LlamaForCausalLM
from promptcache.model.mpt import MptForCausalLM
from promptcache.prompt import Preprocessor, escape_xml, PreprocessorList


# supported models
# gpt2
# aquilla
# bloom
# baichuan

# llama
# llama2
# code llama
# wizardlm
# mpt-chat
# alpaca
# ghatglm
# t5
# falcon
# dolly
#
#
# Aquila (BAAI/Aquila-7B, BAAI/AquilaChat-7B, etc.)
# Baichuan (baichuan-inc/Baichuan-7B, baichuan-inc/Baichuan-13B-Chat, etc.)
# BLOOM (bigscience/bloom, bigscience/bloomz, etc.)
# Falcon (tiiuae/falcon-7b, tiiuae/falcon-40b, tiiuae/falcon-rw-7b, etc.)
# GPT-2 (gpt2, gpt2-xl, etc.)
# GPT BigCode (bigcode/starcoder, bigcode/gpt_bigcode-santacoder, etc.)
# GPT-J (EleutherAI/gpt-j-6b, nomic-ai/gpt4all-j, etc.)
# GPT-NeoX (EleutherAI/gpt-neox-20b, databricks/dolly-v2-12b, stabilityai/stablelm-tuned-alpha-7b, etc.)
# InternLM (internlm/internlm-7b, internlm/internlm-chat-7b, etc.)
# LLaMA & LLaMA-2 (meta-llama/Llama-2-70b-hf, lmsys/vicuna-13b-v1.3, young-geng/koala, openlm-research/open_llama_13b, etc.)
# MPT (mosaicml/mpt-7b, mosaicml/mpt-30b, etc.)
# OPT (facebook/opt-66b, facebook/opt-iml-max-30b, etc.)
# Qwen (Qwen/Qwen-7B, Qwen/Qwen-7B-Chat, etc.)


class FormatConversation(Preprocessor):
    system: (str, str, str)
    user: (str, str)
    assistant: (str, str)

    def __init__(self, system: (str, str, str), user: (str, str), assistant: (str, str)):
        super().__init__()

        self.system = (escape_xml(system[0]), escape_xml(system[1]), escape_xml(system[2]))
        self.user = (escape_xml(user[0]), escape_xml(user[1]))
        self.assistant = (escape_xml(assistant[0]), escape_xml(assistant[1]))

    def __call__(self, prompt: str) -> str:
        replacement_pairs = [
            ("<system>", self.system[0]),
            ("</system>", self.system[1]),
            ("<system/>", self.system[2]),
            ("<user>", self.user[0]),
            ("</user>", self.user[1]),
            ("<assistant>", self.assistant[0]),
            ("</assistant>", self.assistant[1])
        ]

        # remove space before <system>
        prompt = re.sub(r' +<system>', '<system>', prompt)

        for old, new in replacement_pairs:
            prompt = prompt.replace(old, new)

        return prompt


class FormatLlama2Conversation(FormatConversation):

    def __init__(self):
        super().__init__(
            system=("<s>[INST] <<SYS>>\n", "<</SYS>>\n\n", "<s>[INST]"),
            user=(" ", "[/INST]"),
            assistant=(" ", "</s><s>[INST]")
        )


class LanguageModel(abc.ABC):
    name: str
    hf_tokenizer: PreTrainedTokenizer
    hf_model: PreTrainedModel
    stop_token_ids: List[int]
    stop_str: List[str]
    use_full_position_ids: bool = False

    def __init__(self, name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 stop_token_ids: Optional[List[int]] = None, stop_str: Optional[List[str]] = None):
        self.name = name
        self.hf_tokenizer = tokenizer
        self.hf_model = model
        self.stop_token_ids = stop_token_ids if stop_token_ids is not None else [self.eos_token_id]
        self.stop_str = stop_str if stop_str is not None else []

    @abc.abstractmethod
    def get_formatter(self) -> Callable[[str], str]:
        pass

    def get_cache_shape(self) -> Tuple[int, int, int]:
        num_head = self.config.num_attention_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        return self.config.num_hidden_layers, num_head, head_dim

    def store_k_hook(self, k_cache: torch.Tensor) -> torch.Tensor:
        return k_cache

    def store_v_hook(self, v_cache: torch.Tensor) -> torch.Tensor:
        return v_cache

    def read_k_hook(self, k_cache: torch.Tensor) -> torch.Tensor:
        return k_cache

    def read_v_hook(self, v_cache: torch.Tensor) -> torch.Tensor:
        return v_cache

    def __call__(self, **kwargs):
        return self.hf_model(**kwargs)

    def encode(self, text: str) -> List[int]:
        # Warning: this is a hack to remove bos_token
        token_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=False, spaces_between_special_tokens=False)

    @property
    def unk_token(self) -> str:
        return self.hf_tokenizer.unk_token

    @property
    def unk_token_id(self) -> int:
        return self.hf_tokenizer.unk_token_id

    @property
    def eos_token(self) -> str:
        return self.hf_tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        return self.hf_tokenizer.eos_token_id

    @property
    def device(self) -> torch.device:
        return self.hf_model.device

    @property
    def config(self) -> PretrainedConfig:
        return self.hf_model.config


class CodeLlama(LanguageModel):

    def __init__(self, name: str = "codellama/CodeLlama-13b-Instruct-hf", **kwargs):
        tokenizer = CodeLlamaTokenizer.from_pretrained(name)
        model = LlamaForCausalLM.from_pretrained(name, **kwargs)

        self.formatter = FormatConversation(
            system=("<s> [INST] <<SYS>>\n", "<</SYS>>\n\n", "<s> [INST] "),
            user=("", "[/INST]"),
            assistant=("", "</s><s> [INST] "))

        stop_token_ids = [tokenizer.eos_token_id]

        stop_str = ["</s>"]

        super().__init__(name, model, tokenizer, stop_token_ids, stop_str)

    def get_formatter(self) -> Callable[[str], str]:
        return self.formatter


class Llama2(LanguageModel):

    def __init__(self, name: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs):
        tokenizer = LlamaTokenizer.from_pretrained(name)
        model = LlamaForCausalLM.from_pretrained(name, **kwargs)

        self.formatter = FormatConversation(
            system=("<s> [INST] <<SYS>>\n", "<</SYS>>\n\n", "<s> [INST] "),
            user=("", "[/INST]"),
            assistant=("", "</s><s> [INST] "))

        stop_token_ids = [tokenizer.eos_token_id]

        stop_str = ["</s>"]

        super().__init__(name, model, tokenizer, stop_token_ids, stop_str)

    def get_formatter(self) -> Callable[[str], str]:
        return self.formatter


class Falcon(LanguageModel):
    def __init__(self, name="tiiuae/falcon-7b-instruct", **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = FalconForCausalLM.from_pretrained(name, **kwargs)

        def rep(prompt: str) -> str:
            return prompt.replace("\r\n", "\n").replace("\n\n", "\n")

        # name = "falcon",
        # roles = ("User", "Assistant"),
        # messages = [],
        # sep_style = SeparatorStyle.RWKV,
        # sep = "\n",
        # sep2 = "<|endoftext|>",
        # stop_str = "\nUser",

        conv = FormatConversation(
            system=("", "\n\n", ""),
            user=("User: ", "\n\nAssistant:"),
            assistant=(" ", "\n\n"))
        #
        # conv = FormatConversation(
        #     system=("", "\n\n", ""),
        #     user=("<|prompt|>", "<|endoftext|>"),
        #     assistant=("<|answer|>", "<|endoftext|>"))

        self.formatter = PreprocessorList([
            rep, conv
        ])

        stop_token_ids = [0,
                          1,
                          2,
                          3,
                          4,
                          5,
                          6,
                          7,
                          8,
                          9,
                          10,
                          11, ]

        stop_str = ["<|endoftext|>", "\nUser"]

        super().__init__(name, model, tokenizer, stop_token_ids, stop_str)

    def get_formatter(self) -> Callable[[str], str]:
        return self.formatter

    def get_cache_shape(self) -> Tuple[int, int, int]:
        head_dim = self.hf_model.config.hidden_size // self.hf_model.config.num_attention_heads
        return self.hf_model.config.num_hidden_layers, 1, head_dim


class Mpt(LanguageModel):
    def __init__(self, name="mosaicml/mpt-7b-chat", **kwargs):
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        model = MptForCausalLM.from_pretrained(name, max_seq_len=8192,
                                               **kwargs)

        conv = FormatConversation(
            system=("<|im_start|>system\n", "<|im_end|>\n", ""),
            user=("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
            assistant=("", "<|im_end|>\n"))

        self.formatter = conv
        self.use_full_position_ids = True

        stop_token_ids = [50278, 0]
        stop_str = []

        super().__init__(name, model, tokenizer, stop_token_ids, stop_str)

    def get_formatter(self) -> Callable[[str], str]:
        return self.formatter

    # https://huggingface.co/mosaicml/mpt-7b-chat/blob/main/configuration_mpt.py
    def get_cache_shape(self) -> Tuple[int, int, int]:
        head_dim = self.hf_model.config.d_model // self.hf_model.config.n_heads,
        return self.hf_model.config.n_layers, self.hf_model.config.n_heads, head_dim[0]
    #
    # def store_k_hook(self, v_cache: torch.Tensor) -> torch.Tensor:
    #     # batch, n_layers, seq_len, head_dim = v_cache.shape
    #     return v_cache.transpose(2, 3)
    #
    # def read_k_hook(self, v_cache: torch.Tensor) -> torch.Tensor:
    #     return v_cache.transpose(1, 2)
