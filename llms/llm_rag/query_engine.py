from typing import TypedDict, Annotated
from langchain.prompts import ChatPromptTemplate
from core.utils import get_chat_model, format_chat_history


class Result(TypedDict):
    turn_count: Annotated[int, ..., "The turn count based on the user query in <turn_count> tag."]
    state: Annotated[str, ..., "The routing state based on user query in <state> tag."]
    thinking: Annotated[str, ..., "The thinking based on the user query in <thinking> tag."]
    response: Annotated[str, ..., "The response based on the user query in <response> tag."]
    save_event: Annotated[str, ..., "The save event based on the user query in <save event> tag."]


def query_engine(query: str, formatted_prompt: ChatPromptTemplate, model_name: str, model_type: str,
                 temperature: float, streaming: bool, **kwargs):
    llm = get_chat_model()

    chain = formatted_prompt | llm.with_structured_output(Result)

    inputs = {
        'context': kwargs["RAG_context"],
        'messages': kwargs["messages"],
        'user_persona_prompt': kwargs["user_persona_prompt"],
        'user_bio': kwargs["user_bio"],
        'question': query
    }
    response = chain.invoke(inputs)

    return response


async def aquery_engine(query: str, formatted_prompt: ChatPromptTemplate, model_name: str, model_type: str,
                        temperature: float, streaming: bool, **kwargs):
    llm = get_chat_model()

    chain = formatted_prompt | llm.with_structured_output(Result)

    inputs = {
        'context': kwargs["RAG_context"],
        'messages': kwargs["messages"],
        'user_persona_prompt': kwargs["user_persona_prompt"],
        'user_bio': kwargs["user_bio"],
        'question': query
    }
    async for chunk in chain.astream(inputs):
        yield chunk


def demo_invoke():
    from conf.config import system_config, cfg
    from core.state import NationalServiceRAGState

    state = NationalServiceRAGState("duringNS", cfg.prompts.rag_ns_system_during_NS,
                                    cfg.prompts.rag_ns_human_NS, 'benjamin.davis')

    formatted_prompt = state.enter()
    user_input = "What is national service in Singapore"
    context = "aaa"
    response = query_engine(
        query=user_input,
        formatted_prompt=formatted_prompt,
        model_name=system_config['model_name'],
        model_type=system_config['model_type'],
        temperature=system_config['temperature'],
        streaming=system_config['streaming'],
        RAG_context=context,
        messages=format_chat_history([]),
        user_persona_prompt=state.user_persona,
        user_bio=state.user_bio
    )

    print(response)


async def demo_astream():
    from conf.config import system_config, cfg
    from core.state import NationalServiceRAGState

    state = NationalServiceRAGState("duringNS", cfg.prompts.rag_ns_system_during_NS,
                                    cfg.prompts.rag_ns_human_NS, 'benjamin.davis')

    formatted_prompt = state.enter()
    user_input = "What is national service in Singapore"
    context = "aaa"
    async for chunk in aquery_engine(
            query=user_input,
            formatted_prompt=formatted_prompt,
            model_name=system_config['model_name'],
            model_type=system_config['model_type'],
            temperature=system_config['temperature'],
            streaming=system_config['streaming'],
            RAG_context=context,
            messages=format_chat_history([]),
            user_persona_prompt=state.user_persona,
            user_bio=state.user_bio
    ):
        print(chunk)


def main():
    # demo_invoke()

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_astream())
    loop.close()


if __name__ == "__main__":
    main()
