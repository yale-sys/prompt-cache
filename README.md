# Prompt Cache

This is the repository for the [Prompt Cache: Modular Attention Reuse For Low-Latency Inference](https://arxiv.org/abs/2311.04934) paper. This repository includes the implementation and evaluation tools to demonstrate prompt caching technique.

## Getting Started

### Installation

To begin using the Prompt Cache, you need to install the required dependencies:

```bash
pip install -r requirements.txt
```

For evaluations involving `LongBench`, additional installation of `bleurt` from the source is necessary:

```bash
cd ./dependency/bleurt
pip install .
```

### Supported Architectures

The Prompt Cache extends the `transformers` library and is compatible with several Large Language Model (LLM) architectures. The inference engine currently supports:

- **Llama2**: For example, use `meta-llama/Llama-2-7b-chat-hf` or `codellama/CodeLlama-7b-Instruct-hf`.
- **Falcon**: Example configuration is `tiiuae/falcon-7b-instruct`.
- **MPT**: An example model is `mosaicml/mpt-7b-chat-8k`.

Model weights for these architectures are automatically retrieved from the Hugging Face model hub.

### Demo

Explore the capabilities with our demo:

```bash
python demo.py
```

In the demo script `demo.py`, the `use_cache` flag can be toggled to enable or disable the Prompt Cache feature. You are encouraged to modify the script to experiment with different prompts and schemas.

The demo leverages a schema from `./examples/code_generation_game.xml` with the following prompt setup:

```xml
<prompt schema='code-generation-game'>
    <unit.py/>
    <map.py/>
    <player.py/>
    <game.py/>
    <database.py/>
    <user>Create a main entry for the game:</user>
</prompt>
```

### Additional Schemas

Find more examples of schemas in the `./examples` directory, which can serve as templates for creating your custom prompts and schemas. For instance, using the schema found at [/examples/persona_generation.xml](./benchmark/schema_persona_long.xml), you can create a prompt like the following:
```xml

<prompt schema='persona'>
    <age>
        <young-adult/>
    </age>
    <residence>
        <seaside/>
    </residence>
    <education>
        <doctorate/>
    </education>
    <occupation>
        <technology/>
    </occupation>
    <martial-status>
        <married/>
    </martial-status>
    <personality>
        <introverted/>
    </personality>
    <user>
        Introduce about yourself.
    </user>
</prompt>
```

Executing the above prompt will generate the following response,  GPU: NVIDIA RTX 4090 / CPU: Intel i9-13900K

- **Baseline (with KV cache)** (TTFT:  GPU: 286.9ms, CPU: 17917.14 ms)

```
User: Introduce about yourself.
Assistant: Hello! My name is Alex, and I am a 25-year-old woman who lives in a beautiful coastal town. I have recently completed my
 Ph.D. in computer science and am now working as a software engineer at a leading tech company. When I'm not working, I 
 enjoy spending time outdoors, reading books, and trying out new recipes in the kitchen. I am also very passionate about 
 my work and am constantly looking for opportunities to learn and grow in my
```

- **With PromptCache** (TTFT: GPU: 78.2ms, CPU: 511.31ms)

```
User: Introduce about yourself.
Assistant: Hello! My name is Alex, and I'm a 28-year-old software developer who recently moved to the coastal town of Oceanview. I have a doctorate degree in computer science and specialize in artificial intelligence and machine learning. I work as a lead engineer at a tech startup that focuses on developing innovative AI solutions for various industries.
 In my free time, I enjoy exploring the beautiful beaches and hiking trails around Oceanview with my wife, Sarah. We met while we were both pursuing our graduate degrees at Stanford University, and we've been inseparable ever since. We have two adorable kids, Emily and Jack, who keep us busy and bring us endless joy.
 As an introverted person, I often prefer spending time alone or with my close family members, but I also make an effort to connect with others through social events and community activities. I believe in being honest, kind, and respectful towards everyone, regardless of their background or beliefs.
```

## Prompt Markup Language (PML)

Schema and prompts are written in Prompt Markup Language (PML). PML is a simple XML-based language that allows users to define schemas and prompts for the Prompt Cache. 

### Writing schema with PML

```xml
<!-- Schema is a root module. Schema can contain modules and unions -->
<schema name="default">

    <!-- Module name in the same scope must be unique -->
    <module name="preface">

        <!-- 
        Module can be parameterized with <parameter> tag 
        - length: specifies the maximum length of the parameter value. 
        - scaffold: serve as a placeholder for parameter value during cache encoding. If not specified, unk_token will be used as a scaffold.
        -->

        Just some text with parameter:
        <parameter name="param-name" length="5" scaffold="[to be filled later]"/>

        <!-- Modules can be nested -->
        <module name="nested">Nested</module>

        <!--
        There are some helper tags for chat-style LLM prompting. They will be replaced by LLM-specific tokens during cache encoding.
        - <system> tag is used to specify system prompt
        - <user> tag is used to specify user input
        - <assistant> tag is used to specify assistant response 
         -->
        <system>

            <!--
            Union tag is used to specify set of modules where one can be selected. (same offset index)
            - scaffold: serve as a placeholder for selected module during cache encoding. 
            -->
            <union scaffold="system1">
                <module name="system1">System prompt type 1.</module>

                <!--
                Cache can be disabled for specific module by setting cache="false" attribute. 
                In this case, the KV cache for this module will be computed in every request.
                -->
                <module name="system2" cache="false">System prompt type 2.</module>
                <module name="system3">System prompt type 3,
                    with parameter:
                    <parameter name="message" length="10"/>
                </module>
            </union>

            <union>
                <module name="user1" cache="false">User 1 information</module>
                <module name="user2">User 2 information</module>
            </union>
        </system>
    </module>

    <module name="task">
        <union>
            <module name="task-robot-control">Task description 1
                <parameter name="speed" length="5"/>
            </module>
            <module name="task-predict-future">Task description 1</module>
            <module name="task-random">Task description 1</module>
        </union>
    </module>

</schema>

```

### Writing Prompt with PML

```xml

<prompt schema="default">
    <preface param-name="test">
        <!-- Only subset of modules can be selected -->
        <nested/>
        <system3 message="just doing some test"/>
        <task2 val1="test1" val2="test2"/>
        <user2/>
    </preface>
    <task>
        <task-robot-control speed="fast"/>
    </task>
    <user>What will be the next movement of the robot?</user>
    <assistant>It will move forward.</assistant>
    <user>What would be the speed then?</user>
</prompt>
```

#### Compiling Python function into PML

Compiler can be found in `promptcache/compiler.py`. You can use python decorators to compile python functions into PML.

```python
@prompt
def test_prompt(flag, item):
    ""
    "some system message"
    if flag:
        "only if flag is true"
    match item:
        case 1:
            "item is 1"
        case 2:
            "item is 2"
    "some user message"

r = test_prompt(True, 1)

print(r.get_schema())
print(r.get_prompt())
```



### Benchmark and Evaluation

You can run accuracy benchmarks on LongBench with

```bash
python eval_acc.py --help
```

To evaluate the inference time on LongBench, you can run the following command:

```bash
python eval.py --help
```

### Citation
```
@article{gim2023prompt,
  title={Prompt cache: Modular attention reuse for low-latency inference},
  author={Gim, In and Chen, Guojun and Lee, Seung-seob and Sarda, Nikhil and Khandelwal, Anurag and Zhong, Lin},
  journal={Proceedings of Machine learning and systems},
  year={2024}
}
```