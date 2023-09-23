import abc
from typing import Callable, List, Optional
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizer, \
    FalconForCausalLM, PretrainedConfig, PreTrainedModel

from promptcache.prompt import Preprocessor, escape_xml


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

    def __init__(self, name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 stop_token_ids: Optional[List[int]] = None):
        self.name = name
        self.hf_tokenizer = tokenizer
        self.hf_model = model
        self.stop_token_ids = stop_token_ids if stop_token_ids is not None else [self.eos_token_id]

    @abc.abstractmethod
    def get_formatter(self) -> Callable[[str], str]:
        pass

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


class Llama2(LanguageModel):

    def __init__(self, name: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs):
        tokenizer = LlamaTokenizer.from_pretrained(name)
        model = LlamaForCausalLM.from_pretrained(name, **kwargs)

        self.formatter = FormatConversation(
            system=("<s>[INST] <<SYS>>\n", "<</SYS>>\n\n", "<s>[INST]"),
            user=(" ", "[/INST]"),
            assistant=(" ", "</s><s>[INST]"))

        super().__init__(name, model, tokenizer, [tokenizer.eos_token_id])

    def get_formatter(self) -> Callable[[str], str]:
        return self.formatter


class Falcon(LanguageModel):
    def __init__(self, name="tiiuae/falcon-7b-instruct", **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = FalconForCausalLM.from_pretrained(name, **kwargs)

        self.formatter = FormatConversation(
            system=("System: ", "\n", "System :\n"),
            user=("User: ", "\nFalcon:"),
            assistant=(" ", "\n"))

        super().__init__(name, model, tokenizer)

    def get_formatter(self) -> Callable[[str], str]:
        return self.formatter
