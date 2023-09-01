import lxml
import lxml.etree
import gc
from typing import List, Tuple, Union
from tqdm import tqdm
import time
import conversation
import torch
from transformers import (
    AutoTokenizer,

)

import schema
import cache_engine

model_path = "meta-llama/Llama-2-13b-chat-hf"

chat_layout_raw = """
<system>
</system>

<user>
</user>

<assistant>
</assistant>

<user>
</user>

<assistant>
</assistant>

"""

# make modules nested
# make conversation tags (system, user, assistant). These cannot be nested


layouts_raw = """
<schema name="default">

    <module name="layout1">
        Just some text with parameter: <parameter name="val" length="5" scaffold="sdsd"/>

        <module> Anonymous prompt 1 </module>
        <module name="task1"> Task module 1 </module>
        
        <system>
            <union scaffold="system1">
                <module name="system1"> System prompt type 1. </module>
                <module name="system2" cache="false"> System prompt type 2. </module>
                <module name="system3"> System prompt type 3,
                    with parameter: <parameter name="val" length="10" />
                </module>
            </union>
            
            <module name="task2"> Task module 2 with two parameters: 
                <parameter name="val1" length="10" /> and <parameter name="val2" length="10" />
            </module>
            <union>
                <module name="user1"> User 1 information </module>
                <module name="user2"> User 2 information </module>
            </union>
        </system>
    </module>
    
</schema>
"""

user_prompt_raw = """
<prompt schema="default">
<layout1 val="test">
    <task1 />
    <system3 val="just doing some test" />
    <task2 val1="test1" val2="test2" />
    <user2/>
</layout1>    
<user>what is this thing</user>
</prompt>
"""


# cached conversation template


def main():
    # model = LlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    layout = schema.Schema(layouts_raw, tokenizer)

    print(layout.name)
    print(layout)


if __name__ == "__main__":
    main()
