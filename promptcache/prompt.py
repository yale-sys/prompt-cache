from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Union, List, Any, Optional
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


def apply_preproc(text: str, preprocessors: List[Preprocessor] = None) -> str:
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


class PreprocessorList(Preprocessor):
    """This class is used to preprocess the prompt before it is passed to the model."""
    pre: List[Preprocessor]

    def __init__(self, pre: List[Preprocessor]):
        self.pre = pre

        super().__init__()

    def __call__(self, prompt: str) -> str:
        for p in self.pre:
            prompt = p(prompt)
        return prompt


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
        if root.text is not None and len(root.text.strip()) > 0:
            raise ValueError("Module reference cannot have text")

        for e in root:
            self.modules.append(ModuleRef(e))
            if e.tail is not None and len(e.tail.strip()) > 0:
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

    preproc: List[Preprocessor]

    def __init__(self, spec: Union[str, lxml.etree.Element], preproc: Optional[List[Preprocessor]] = None):

        super().__init__()
        self.args = []
        self.preproc = preproc if preproc is not None else []
        self.text = ""
        if type(spec) == str:

            for p in self.preproc:
                spec = p(spec)

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

        self.text = self.text.strip()

    def add_text(self, text: str):
        for p in self.preproc:
            text = p(text)

        self.text += text

    def __repr__(self) -> str:
        r = f"Schema: @{self.name}"
        for m in self.modules:
            r += '\n' + repr_indent(m)
        r += '\nText: ' + repr(self.text)
        return r
