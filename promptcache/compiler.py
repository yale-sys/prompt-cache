import ast
import inspect
import astor
from functools import wraps


class SchemaBuilder:
    schema: list[str | tuple[bool, str], tuple[str, dict]]
    schema_name: str | None

    def __init__(self):
        self.schema = []
        self.schema_name = None

    def text(self, value: str):
        self.schema.append(value)

    def cond(self, cond: bool, _):
        value = self.schema.pop()
        self.schema.append((cond, value))

    def union(self, target: str, values: dict):
        self.schema.append((target, values))

    def get_schema(self) -> str:

        uid = 0
        output = ""
        for item in self.schema:
            match item:
                case str(val):
                    output += val + "\n"
                case (bool(), str(val)):
                    output += f"<module name=\"m_{uid}\">{val}</module>"
                    uid += 1
                case (str(), dict(d)):
                    output += "<union>"
                    for cond, val in d.items():
                        output += f"<module name=\"m_{uid}\">{val}</module>"
                        uid += 1
                    output += "</union>"

        self.schema_name = f'{abs(hash(output)):x}'

        output = f"<schema name=\"{self.schema_name}\">" + output
        output += "</schema>"
        return output

    def get_prompt(self) -> str:

        if self.schema_name is None:
            self.get_schema()

        output = f"<prompt schema=\"{self.schema_name}\">"
        uid = 0
        for item in self.schema:
            match item:
                case (bool(cond), str()):
                    if cond:
                        output += f"<m_{uid}/>"
                    uid += 1
                case (str(target), dict(d)):
                    for cond, val in d.items():
                        if target == cond:
                            output += f"<m_{uid}/>"
                        uid += 1
                case _:
                    ...
        output += "</prompt>"
        return output


class CodeTransformer(ast.NodeTransformer):

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Safely format the string
            return ast.parse(f'builder.text({repr(node.value.value)})').body[0]
        return node

    def visit_If(self, node):
        condition = astor.to_source(node.test).strip()
        then_body = [self.visit(n) for n in node.body]
        content = astor.to_source(then_body[0]).strip()
        return ast.parse(f'builder.cond({condition}, {content})').body[0]

    def visit_Match(self, node):
        subject = astor.to_source(node.subject).strip()
        cases = {}
        for case in node.cases:
            key = astor.to_source(case.pattern.value).strip()
            value = astor.to_source(case.body[0]).strip()
            cases[key] = value
        cases_str = ", ".join([f'{key}: {value}' for key, value in cases.items()])
        return ast.parse(f'builder.union({subject}, {{{cases_str}}})').body[0]


cached_compiled_funcs = dict()


def prompt(func):
    #
    @wraps(func)
    def wrapper(*args, **kwargs):
        global cached_compiled_funcs

        if func in cached_compiled_funcs:
            print('using cached')
            return cached_compiled_funcs[func](*args, **kwargs)

        source_code = inspect.getsource(func).splitlines()
        # strip decorators from the source itself
        source_code = "\n".join(line for line in source_code if not line.startswith('@'))

        tree = ast.parse(source_code)

        # Insert 'env = Env()' at the start of the function body
        env_init = ast.parse("builder = SchemaBuilder()").body[0]
        tree.body[0].body.insert(0, env_init)

        # Ensure the transformed function returns 'env'
        tree.body[0].body.append(ast.parse("return builder").body[0])

        transformer = CodeTransformer()
        new_tree = transformer.visit(tree)

        # for debugging
        new_source_code = astor.to_source(new_tree)

        local_vars = {}
        exec(compile(new_tree, filename="<ast>", mode="exec"), func.__globals__, local_vars)
        modified_func = local_vars[func.__name__]
        cached_compiled_funcs[func] = modified_func
        return modified_func(*args, **kwargs)

    return wrapper
