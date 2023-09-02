# PromptCache

Modular and structured prompt caching for low-latency LLM inference

### Evaluation tasks

#### 1. Personalization

LLM is widely used for simulating in-person interactions, each with unique personas.
Personas are a combination of unique traits from a set of traits, e.g., select one from {chinese, korean, japanese} and
one from {young, old}.

`PromptCache` allows for efficient caching of prompts for each persona combination, without having to store all possible
combinations.

#### 2. Code generation

Code generation, e.g., generating python code from natural language description, and
autocompleting the code based on the context, is a common task in LLM.

Since the code itself is a structured data, `PromptCache` can be used to cache the code generation prompts,

#### 3. Parameterized prompts

Long task prompts typically involves a parameterizable template.



#### 4. Long contexts

Long contexts about documents, videos, or images are often used in LLM.



### Prompt schema

#### Example

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

### Prompt

#### Example

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