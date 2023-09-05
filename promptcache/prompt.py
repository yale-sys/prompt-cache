from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Union, List, Any
from dataclasses import dataclass

import lxml
import lxml.etree

import xml.sax.saxutils


def repr_indent(obj: Any, indent: int = 1) -> str:
    return '\n'.join([indent * '\t' + s for s in repr(obj).split('\n')])


def read_file(filename: str, preprocessors: List[Preprocessor] = None) -> str:
    with open(filename, 'r') as f:

        text = f.read()

        if preprocessors is not None:
            for p in preprocessors:
                text = p(text)

        return text


def escape_xml(data):
    return xml.sax.saxutils.escape(data, entities={
        "'": "&apos;",
        "\"": "&quot;"
    })


# Replaces multiple leading and trailing whitespaces with a single space.
def compact_surrounding_spaces(text: str) -> str:
    return re.sub(r'^\s+|\s+$', ' ', text)


def compact_spaces(text: str) -> str:
    return ' '.join(text.split())


class Preprocessor(ABC):
    """This class is used to preprocess the prompt before it is passed to the model."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError


class CompactSpaces(Preprocessor):
    """This class is used to remove all leading and trailing whitespaces."""
    only_surrounding: bool

    def __init__(self, only_surrounding: bool = False):
        super().__init__()
        self.only_surrounding = only_surrounding

    def __call__(self, prompt: str) -> str:

        if self.only_surrounding:
            return compact_surrounding_spaces(prompt)
        else:
            return compact_spaces(prompt)


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


class ModuleRef:
    """This class is used to represent a module reference in the prompt."""
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

    def __repr__(self) -> str:

        args = " ".join([f"{arg.name}={repr(arg.value)}" for arg in self.args])
        if len(args) > 0:
            r = f"@{self.name}({args})"
        else:
            r = f"@{self.name}"

        for m in self.modules:
            r += '\n' + repr_indent(m)
        return r


@dataclass
class Argument:
    name: str
    value: str


class Prompt(ModuleRef):
    """This class is used to represent a prompt."""
    schema: str
    text: str

    def __init__(self, spec: Union[str, lxml.etree.Element]):

        super().__init__()
        self.args = []

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

            self.text = compact_surrounding_spaces(text)

        else:
            if root.text is not None and len(root.text.strip()) > 0:
                raise ValueError("Prompt cannot have leading text")

            tail = False
            for e in root:
                if tail:
                    raise ValueError("Prompt cannot have text between module references")

                self.modules.append(ModuleRef(e))

                if e.tail is not None:
                    self.text = compact_surrounding_spaces(e.tail)

    def __repr__(self) -> str:
        r = f"Schema: @{self.name}"
        for m in self.modules:
            r += '\n' + repr_indent(m)
        r += '\nText: ' + repr(self.text)
        return r
