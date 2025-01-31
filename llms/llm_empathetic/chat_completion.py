from core.utils import extract_empathetic_content, dict2xml
from typing import Dict, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from core.utils import get_chat_model, dict2xml, format_chat_history


class Result(TypedDict):
    turn_count: Annotated[int, ..., "The turn count based on the user query in <turn_count> tag."]
    state: Annotated[str, ..., "The routing state based on user query in <state> tag."]
    thinking: Annotated[str, ..., "The thinking based on the user query in <thinking> tag."]
    response: Annotated[str, ..., "The response based on the user query in <response> tag."]
    save_event: Annotated[str, ..., "The save event based on the user query in <save event> tag."]


class EmpatheticLLM:
    def __init__(self, args):
        self.chat_model = get_chat_model()

    def chat(self, formatted_empathetic_prompt, user_input: str,
             user_persona_prompt: str, user_bio: str, chat_history: Optional[list] = None):
        if chat_history is None:
            chat_history = []

        formatted_history = format_chat_history(chat_history)
        formatted_history.append(HumanMessage(content=user_input))

        chain = formatted_empathetic_prompt | self.chat_model.with_structured_output(Result)
        inputs = {
            'messages': formatted_history,
            'user_persona_prompt': user_persona_prompt,
            'user_bio': user_bio
        }
        result = chain.invoke(inputs)
        text = dict2xml(result)
        return result, text

    async def achat(self, formatted_empathetic_prompt, user_input: str,
                    user_persona_prompt: str, user_bio: str, chat_history: Optional[list] = None):
        if chat_history is None:
            chat_history = []

        formatted_history = format_chat_history(chat_history)
        formatted_history.append(HumanMessage(content=user_input))

        chain = formatted_empathetic_prompt | self.chat_model.with_structured_output(Result)
        inputs = {
            'messages': formatted_history,
            'user_persona_prompt': user_persona_prompt,
            'user_bio': user_bio
        }

        last_chunk = {}
        for chunk in chain.stream(inputs):
            last_chunk = chunk
            yield chunk, ""

        result = last_chunk
        text = dict2xml(result)

        yield result, text


def demo_invoke():
    from conf.config import system_config, cfg
    from core.state import EmpatheticState

    state = EmpatheticState("duringNS", cfg.prompts.empathetic_during_NS, 'benjamin.davis')
    formatted_prompt = state.enter()
    user_input = 'Hello'

    llm = EmpatheticLLM(system_config)

    reply, text = llm.chat(formatted_prompt, user_input, state.user_persona, state.user_bio, [])
    print(reply)


async def demo_astream():
    from conf.config import system_config, cfg
    from core.state import EmpatheticState

    state = EmpatheticState("duringNS", cfg.prompts.empathetic_during_NS, 'benjamin.davis')
    formatted_prompt = state.enter()
    user_input = 'Hello'

    llm = EmpatheticLLM(system_config)

    async for reply, reply_text in llm.achat(formatted_prompt, user_input,
                                             state.user_persona, state.user_bio, []):
        print(reply)


def main():
    demo_invoke()

    # import asyncio
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(demo_astream())
    # loop.close()


if __name__ == "__main__":
    main()
