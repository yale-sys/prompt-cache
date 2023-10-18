# Type hoisting
from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
import re

import lxml
import lxml.etree

from typing import List, Union, cast, Any, Optional

from .model import LanguageModel
from .prompt import compact_surrounding_spaces


def trim_with_padding(text: str, padding: int = 1) -> str:
    pad_str = ' ' * padding
    return pad_str + text.strip() + pad_str


def is_valid_xml_element_name(name: str) -> bool:
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_\-.]*$"
    return bool(re.fullmatch(pattern, name))


def repr_indent(obj: Any, indent: int = 1) -> str:
    return '\n'.join([indent * '\t' + s for s in repr(obj).split('\n')])


class Path:
    path: List[str]

    def __init__(self, path: Optional[Union[str, List[str]]] = None):

        if path is None:
            path = []

        if type(path) == str:
            if '/' in path:
                path = [s.strip() for s in path.split('/')]
            elif len(path) > 0:
                path = [path]
            else:
                path = []

        self.path = path

    def __len__(self):
        return len(self.path)

    def __str__(self):
        return '/'.join(self.path)

    def __repr__(self):
        return str(self)

    @property
    def is_root(self) -> bool:
        return len(self.path) == 0

    @property
    def head(self) -> Optional[str]:
        return None if self.is_empty else self.path[0]

    @property
    def is_empty(self) -> bool:
        return len(self.path) == 0

    @property
    def next(self) -> Path:
        return Path(self.path[1:])


class Element(ABC):
    name: Union[None, str]
    offset: int

    def __init__(self, offset: int, name: Optional[str] = None):
        self.name = name
        self.offset = offset

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self):
        return f"[{self.offset}:{self.offset + len(self)}]"

    @abstractmethod
    def token_ids(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def position_ids(self) -> List[int]:
        raise NotImplementedError


class Parameter(Element):
    length: int
    placeholder_token: int
    _token_ids: List[int]
    _position_ids: List[int]

    def __init__(self,
                 offset: int,
                 spec: lxml.etree.Element,
                 lm: LanguageModel):
        super().__init__(offset)

        self.placeholder_token = lm.unk_token_id
        self._process(spec, lm)

    def _process(self, root: lxml.etree.Element, lm: LanguageModel):

        assert root.tag == "parameter"

        if "name" not in root.attrib:
            raise ValueError(f'Parameter name is missing')

        if "length" not in root.attrib:
            raise ValueError(f'Parameter length (in tokens) is missing')

        if not is_valid_xml_element_name(root.attrib["name"]):
            raise ValueError(f'Parameter name {root.attrib["name"]} is not valid')

        self.name = root.attrib["name"]
        self.length = int(root.attrib['length'])

        self._token_ids = []
        self._position_ids = list(range(self.offset, self.offset + self.length))

        if "scaffold" in root.attrib:

            self._token_ids = lm.encode(root.attrib["scaffold"])

            if len(self._token_ids) > self.length:
                raise ValueError(f'Scaffold for parameter {self.name} is too long')

        self._token_ids += [self.placeholder_token] * (self.length - len(self._token_ids))

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return super().__repr__() + f" Parameter @{self.name}"

    def token_ids(self) -> List[int]:
        return self._token_ids

    def position_ids(self) -> List[int]:
        return self._position_ids


class TokenSequence(Element):
    text: str
    _token_ids: List[int]
    _position_ids: List[int]

    def __init__(self, offset: int, text: str, lm: LanguageModel, max_tokens: Optional[int] = None):
        super().__init__(offset)

        self.text = text
        self._token_ids = lm.encode(text)

        if max_tokens is not None:
            self._token_ids = self._token_ids[:max_tokens // 2] + self._token_ids[-max_tokens // 2:]

        self._position_ids = list(range(self.offset, self.offset + len(self._token_ids)))

    def __len__(self) -> int:
        return len(self._token_ids)

    def __repr__(self) -> str:
        return super().__repr__() + f" Text: {repr(self.text)}" + "\n\t" + repr(self.token_ids())

    def token_ids(self) -> List[int]:
        return self._token_ids

    def position_ids(self) -> List[int]:
        return self._position_ids


class UnionModule(Element):
    modules: List[Module]
    length: int
    scaffold_name: Optional[str]

    def __init__(self, offset, spec: lxml.etree.Element, lm: LanguageModel, max_tokens: Optional[int] = None):

        super().__init__(offset)

        self.modules = []
        self.length = 0
        self.scaffold_name = None

        self._process(spec, lm, max_tokens)

    def _process(self, root: lxml.etree.Element, lm: LanguageModel, max_tokens: Optional[int] = None):

        assert root.tag == "union"

        max_len = 0

        for e in root:
            if e.tag != "module":
                raise ValueError("Only <module> tags are allowed in union")

            module = Module(self.offset, e, lm, max_tokens=max_tokens)
            self.modules.append(module)
            max_len = max(max_len, len(module))

        if "scaffold" in root.attrib:
            scaffold = root.attrib["scaffold"]

            if self.select(scaffold) is None:
                raise ValueError(f"Union scaffold {scaffold} is not found in union")
            self.scaffold_name = scaffold

        # if scaffold is empty, set first element as scaffold
        # else:
        #     self.scaffold_name = self.modules[0].name

        self.length = max_len

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        r = super().__repr__() + " Union"
        for m in self.modules:
            r += '\n' + repr_indent(m)
        return r

    def token_ids(self) -> List[int]:
        raise ValueError("Cannot get token_ids() on union. Try again on its scaffold")

    def position_ids(self) -> List[int]:
        raise ValueError("Cannot get position_ids() on union. Try again on its scaffold")

    def select(self, path: Union[str, Path]) -> Optional[Module]:
        #
        if path is None:
            return None

        if type(path) == str:
            path = Path(path)

        if path.is_root:
            raise ValueError("Cannot select root of union")

        for m in self.modules:
            if m.name == path.head:
                if len(path) == 1:
                    return m
                else:
                    return m.select(path.next)
        return None


class Module(Element):
    children: List[Element]
    length: int
    cache: bool
    _is_root: bool
    _contains_union: bool

    def __init__(self,
                 offset: int,
                 spec: Union[str, lxml.etree.Element],
                 lm: LanguageModel,
                 is_root: bool = False,
                 max_tokens: Optional[int] = None):

        super().__init__(offset)

        self.children = []
        self.length = 0
        self.cache = True  # whether to do cache (true by default)
        self._is_root = is_root
        self._contains_union = False

        if type(spec) == str:
            parser = lxml.etree.XMLParser(recover=True)
            spec = lxml.etree.fromstring(spec, parser=parser)

        self._process(spec, lm, max_tokens)

    def _process(self, root: lxml.etree.Element, lm: LanguageModel, max_tokens: Optional[int] = None):

        if self._is_root:
            assert root.tag == "schema"
        else:
            assert root.tag == "module"

        if "name" not in root.attrib:
            raise ValueError("Module name is missing")

        if not is_valid_xml_element_name(root.attrib["name"]):
            raise ValueError(f'Module name {root.attrib["name"]} is not valid')

        if not self._is_root and "cache" in root.attrib:
            self.cache = root.attrib["cache"] == "true"

        self.name = root.attrib["name"]

        offset = self.offset
        self.children = []

        if "src" in root.attrib:
            src_path = pathlib.Path(root.attrib["src"])

            # check if file exists
            if not src_path.exists():
                raise ValueError(f"Module source file {src_path} does not exist")

            text = compact_surrounding_spaces(src_path.read_text())

            if len(text) > 0:
                seq = TokenSequence(offset, text, lm, max_tokens=max_tokens)
                self.children.append(seq)
                offset += len(seq)

        # prefix text
        if root.text is not None:
            text = compact_surrounding_spaces(root.text)
            if len(text) > 0:
                seq = TokenSequence(offset, text, lm, max_tokens=max_tokens)
                self.children.append(seq)
                offset += len(seq)

        for e in root:
            match e.tag:
                case "module":
                    m = Module(offset, e, lm, max_tokens=max_tokens)
                    self._contains_union = self._contains_union or m._contains_union

                    # check namespace conflicts
                    submodule_names = [c.name for c in self.modules()]
                    if m.name in submodule_names:
                        raise ValueError(f"Module {m.name} is already defined")

                case "union":
                    m = UnionModule(offset, e, lm, max_tokens=max_tokens)
                    self._contains_union = True
                    submodule_names = [c.name for c in self.modules()]
                    for c in m.modules:
                        if c.name in submodule_names:
                            raise ValueError(f"Module {c.name} is already defined")

                case "parameter":
                    if self._is_root:
                        raise ValueError("Parameters are not allowed in schema")

                    m = Parameter(offset, e, lm)

                    parameter_names = [c.name for c in self.parameters()]
                    if m.name in parameter_names:
                        raise ValueError(f"Parameter {m.name} is already defined")

                case _:
                    m = TokenSequence(offset, lxml.etree.tostring(e), lm, max_tokens=max_tokens)

            self.children.append(m)
            offset += len(m)

            # process tailing text
            if e.tail is not None:
                text = compact_surrounding_spaces(e.tail)
                if len(text) > 0:
                    seq = TokenSequence(offset, text, lm, max_tokens=max_tokens)
                    self.children.append(seq)
                    offset += len(seq)

        self.length = offset - self.offset

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        r = super().__repr__() + f" Module @{self.name}"
        for m in self.children:
            r += '\n' + repr_indent(m)
        return r

    def token_ids(self) -> List[int]:
        if self.contains_union():
            raise ValueError("Cannot get token_ids() on module that contains union. Try again on its scaffold")
        else:
            return [e for c in self.children for e in c.token_ids()]

    def position_ids(self) -> List[int]:
        if self.contains_union():
            raise ValueError("Cannot get position_ids() on module that contains union. Try again on its scaffold")
        else:
            return [e for c in self.children for e in c.position_ids()]

    def get_scaffold(self, *paths: Path) -> Scaffold:
        return Scaffold(self, *paths)

    def select(self, path: Union[str, Path]) -> Optional[Module]:
        if type(path) == str:
            path = Path(path)

        if path.is_root:
            return self

        for p in self.modules():
            if p.name == path.head:
                if len(path) == 1:
                    return p
                else:
                    return p.select(path.next)
        return None

    def modules(self) -> List[Module]:
        modules = []
        for c in self.children:
            if type(c) == Module:
                modules.append(c)
            elif type(c) == UnionModule:
                c = cast(UnionModule, c)
                for m in c.modules:
                    modules.append(m)
        return modules

    def parameters(self) -> List[Parameter]:
        return [cast(Parameter, c) for c in self.children if type(c) == Parameter]

    def token_sequences(self) -> List[TokenSequence]:
        return [cast(TokenSequence, c) for c in self.children if type(c) == TokenSequence]

    def contains_union(self) -> bool:
        return self._contains_union


# Scaffold is a special module that does not contain unions
class Scaffold(Element):
    module: Module
    children: List[Element]

    def __init__(self, module: Module, *paths: Path):
        super().__init__(module.offset, module.name)
        self.module = module

        self._process(*paths)

    def _process(self, *paths: Path):

        self.children = []

        for e in self.module.children:
            if type(e) == UnionModule:
                union = cast(UnionModule, e)
                rel_paths = [n for n in paths if union.select(n.head)]
                unique_names = list(set([n.head for n in rel_paths]))

                if len(unique_names) > 1:
                    raise ValueError(f"Union cannot have multiple names in scaffold")

                if rel_paths is None or len(rel_paths) == 0:
                    selected_module_name = union.scaffold_name
                else:
                    selected_module_name = unique_names[0]

                if selected_module_name is None:
                    continue

                selected_module = union.select(selected_module_name)
                scaffold = Scaffold(selected_module, *[n.next for n in rel_paths])

                self.children.append(scaffold)

            elif type(e) == Module:
                module = cast(Module, e)
                scaffold = Scaffold(module, *[n.next for n in paths if n.head == module.name])
                self.children.append(scaffold)

            else:
                self.children.append(e)

    def __len__(self):
        return self.module.length

    def __repr__(self) -> str:
        r = super().__repr__() + f" Scaffold @{self.name}"
        for m in self.children:
            r += '\n' + repr_indent(m)
        return r

    def token_ids(self) -> List[int]:
        return [e for c in self.children for e in c.token_ids()]

    def position_ids(self) -> List[int]:
        return [e for c in self.children for e in c.position_ids()]

    def select(self, path: Union[str, Path]) -> Optional[Scaffold]:
        if type(path) == str:
            path = Path(path)

        if path.is_root:
            return self

        for e in self.children:
            if type(e) == Scaffold:
                scaffold = cast(Scaffold, e)
                if scaffold.name == path.head:
                    if len(path) == 1:
                        return scaffold
                    else:
                        return scaffold.select(path.next)
        return None

    # return all token sequences in this scaffold
    def all_token_sequences(self) -> List[TokenSequence]:
        ret = []
        for e in self.children:
            if type(e) == Scaffold:
                ret += cast(Scaffold, e).all_token_sequences()
            elif type(e) == TokenSequence:
                ret.append(cast(TokenSequence, e))
        return ret


# Schema is a root module that cannot contain parameters
class Schema(Module):
    lm: LanguageModel

    def __init__(self, spec: Union[str, lxml.etree.Element], lm: LanguageModel, max_tokens: Optional[int] = None):
        super().__init__(0, spec, lm, is_root=True, max_tokens=max_tokens)

        self.lm = lm
