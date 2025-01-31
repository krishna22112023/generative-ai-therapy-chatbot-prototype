from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
)
from typing import List, Union, Dict, Any
import re


class PromptLoader:
    def __init__(self):
        self.template = ""
        self.input_variables = []
        self.input_types = {}
        self.messages = []

    def load_template(self, template: str):
        self.template = template
        self.input_variables = self._detect_variables(template)
        return self

    def _detect_variables(self, template: str) -> List[str]:
        return re.findall(r"\{(\w+)\}", str(template))

    def add_system_message(self, template: str, variables: List[str] = None):
        if variables is None:
            variables = self._detect_variables(template)
        self.messages.append(
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(input_variables=variables, template=template)
            )
        )
        self.input_variables.extend(variables)
        return self

    def add_human_message(self, template: str, variables: List[str] = None):
        if variables is None:
            variables = self._detect_variables(template)
        self.messages.append(
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(input_variables=variables, template=template)
            )
        )
        self.input_variables.extend(variables)
        return self

    def add_messages_placeholder(self, variable_name: str):
        self.messages.append(MessagesPlaceholder(variable_name=variable_name))
        self.input_variables.append(variable_name)
        self.input_types[variable_name] = List[
            Union[AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage]
        ]
        return self

    def set_input_type(self, variable: str, type_hint: Any):
        self.input_types[variable] = type_hint
        return self

    def create_prompt(self) -> ChatPromptTemplate:
        self.input_variables = list(set(self.input_variables))  # Remove duplicates
        return ChatPromptTemplate(
            input_variables=self.input_variables,
            input_types=self.input_types,
            messages=self.messages,
        )


def create_llm_router_prompt(template: str, prompt_args: list = None):
    return (
        PromptLoader()
        .load_template(str(template))
        .add_system_message(template, prompt_args)
        .add_messages_placeholder("messages")
        .set_input_type(
            "messages",
            List[
                Union[
                    AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage
                ]
            ],
        )
        .create_prompt()
    )
