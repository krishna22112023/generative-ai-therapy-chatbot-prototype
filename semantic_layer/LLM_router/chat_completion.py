from core.utils import extract_router_states
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from conf.config import system_config


class RouterLLM:
    def __init__(self, args):
        if args is None:
            args = system_config

        self.chat_model = self._initialize_chat_model(
            args.get("model_name"),args.get("model_type"), args.get("temperature"), args.get("streaming")
        )

    def _initialize_chat_model(
        self, model_name: str,model_type:str, temperature: float, streaming: bool
    ) -> ChatOpenAI:
        if model_type == 'openai':
            if 'gpt' in model_name:
                return ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming
                )
            else:
                raise ValueError(f"Invalid openai model name {model_name}")
        elif model_type == 'groq':
            if 'llama' in model_name or 'mixtral' in model_name:
                return ChatGroq(model=model_name,temperature=temperature)
            else:
                raise ValueError(f"Invalid groq model name {model_name}")
        else : 
            raise NotImplementedError(f"{model_type} model type not implemented.")

    def chat(
        self,
        formatted_routing_prompt,
        user_input: str,
        chat_history: Optional[list] = None,
    ):
        """
        Generate a response using the Singlish AI Kaki.

        Args:
            user_input (str): The latest user message.
            user_persona_prompt (str): Information about the user's persona.
            user_bio (str): User's biography.
            chat_history (list, optional): List of previous messages.
        """
        if chat_history is None:
            chat_history = []

        chat_history.append(HumanMessage(content=user_input))
        response = self.chat_model(
            formatted_routing_prompt.format_messages(messages=chat_history)
        )
        structured_response = extract_router_states(response.content)
        return structured_response, response.content
