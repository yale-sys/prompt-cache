from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Union, List
from dataclasses import dataclass

import lxml
import lxml.etree


# Replaces multiple leading and trailing whitespaces with a single space.
def compact_spaces(text: str) -> str:
    return re.sub(r'^\s+|\s+$', ' ', text)


class Preprocessor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError


class ChatPreprocessor(Preprocessor):
    system: (str, str, str)
    user: (str, str)
    assistant: (str, str)

    def __init__(self, system: (str, str, str), user: (str, str), assistant: (str, str)):
        super().__init__()

        self.system = system
        self.user = user
        self.assistant = assistant

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

        for old, new in replacement_pairs:
            prompt = prompt.replace(old, new)

        return prompt


class Llama2ChatPreprocessor(ChatPreprocessor):

    def __init__(self):
        super().__init__(
            system=("<s>[INST] <<SYS>>\n", "<</SYS>>\n\n", "<s>[INST]"),
            user=(" ", "[/INST]"),
            assistant=(" ", "</s><s>[INST]")
        )


class ModuleRef:
    name: str
    args: List[Argument]
    modules: List[ModuleRef]

    def __init__(self, spec: lxml.etree.Element = None):
        if spec is not None:
            self._process(spec)

    def _process(self, root: lxml.etree.Element):
        self.name = root.tag
        self.args = [Argument(name, value) for name, value in root.attrib.items()]

        self.modules = []

        # leading text
        if root.text is not None:
            raise ValueError("Module reference cannot have text")

        for e in root:
            self.modules.append(ModuleRef(e))
            if e.tail is not None:
                raise ValueError("Module reference cannot have text")


@dataclass
class Argument:
    name: str
    value: str


class Prompt(ModuleRef):
    schema: str
    text: str

    def __init__(self, spec: Union[str, lxml.etree.Element]):

        super().__init__()

        if type(spec) == str:
            parser = lxml.etree.XMLParser(recover=True)
            spec = lxml.etree.fromstring(spec, parser=parser)

        self._process(spec)

    def _process(self, root: lxml.etree.Element):

        assert root.tag == "prompt"

        if "schema" in root.attrib:
            self.schema = root.attrib["schema"]
        else:
            self.schema = ""  # empty schema

        self.name = self.schema
        self.modules = []

        # text only prompt
        if len(root) == 0:
            text = root.text
            if text is None:
                raise ValueError("Prompt cannot be empty")

            self.text = compact_spaces(text)

        else:
            if root.text is not None and len(root.text.strip()) > 0:
                raise ValueError("Prompt cannot have leading text")

            tail = False
            for e in root:
                if tail:
                    raise ValueError("Prompt cannot have text between module references")

                self.modules.append(ModuleRef(e))

                if e.tail is not None:
                    self.text = compact_spaces(e.tail)
