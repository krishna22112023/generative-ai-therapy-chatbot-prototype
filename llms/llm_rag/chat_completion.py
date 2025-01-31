from typing import Optional
from llms.llm_rag.query_engine import query_engine, aquery_engine
from langchain.schema import HumanMessage
from core.utils import format_chat_history, dict2xml


class NationalServiceRAGLLM:
    def __init__(self, args):
        self.model_name = args["model_name"]
        self.temperature = args["temperature"]
        self.streaming = args["streaming"]
        self.model_type = args["model_type"]

    def structured_chat(self, formatted_ns_rag_prompt, search_results: list[dict], user_input: str,
                        user_persona_prompt: str, user_bio: str, chat_history: Optional[list] = None):
        if chat_history is None:
            chat_history = []

        context = "\n\n".join([f"Document {i + 1}:\n{result['content']}" for i, result in enumerate(search_results)])

        formatted_history = format_chat_history(chat_history)
        formatted_history.append(HumanMessage(content=user_input))

        result = query_engine(
            query=user_input,
            formatted_prompt=formatted_ns_rag_prompt,
            model_name=self.model_name,
            model_type=self.model_type,
            temperature=self.temperature,
            streaming=self.streaming,
            RAG_context=context,
            messages=formatted_history,
            user_persona_prompt=user_persona_prompt,
            user_bio=user_bio
        )

        text = dict2xml(result)
        return result, text

    async def astructured_chat(self, formatted_ns_rag_prompt, search_results: list[dict], user_input: str,
                               user_persona_prompt: str, user_bio: str, chat_history: Optional[list] = None):
        if chat_history is None:
            chat_history = []

        context = "\n\n".join([f"Document {i + 1}:\n{result['content']}" for i, result in enumerate(search_results)])

        formatted_history = format_chat_history(chat_history)
        formatted_history.append(HumanMessage(content=user_input))

        last_chunk = {}
        async for chunk in aquery_engine(
                query=user_input,
                formatted_prompt=formatted_ns_rag_prompt,
                model_name=self.model_name,
                model_type=self.model_type,
                temperature=self.temperature,
                streaming=self.streaming,
                RAG_context=context,
                messages=formatted_history,
                user_persona_prompt=user_persona_prompt,
                user_bio=user_bio
        ):
            last_chunk = chunk
            yield chunk, ""

        result = last_chunk
        text = dict2xml(result)
        yield result, text


def demo_invoke():
    from conf.config import system_config, cfg
    from core.state import NationalServiceRAGState

    state = NationalServiceRAGState("duringNS", cfg.prompts.rag_ns_system_during_NS,
                                    cfg.prompts.rag_ns_human_NS, 'benjamin.davis')
    formatted_prompt = state.enter()
    user_input = "What is national service in Singapore"
    search_results = [{'content': "execise!"}]

    llm = NationalServiceRAGLLM(system_config)
    result, text = llm.structured_chat(
        formatted_prompt,
        search_results,
        user_input,
        state.user_persona,
        state.user_bio,
        [],
    )
    print(result)


async def demo_astream():
    from conf.config import system_config, cfg
    from core.state import NationalServiceRAGState

    state = NationalServiceRAGState("duringNS", cfg.prompts.rag_ns_system_during_NS,
                                    cfg.prompts.rag_ns_human_NS, 'benjamin.davis')
    formatted_prompt = state.enter()
    user_input = "What is national service in Singapore"
    search_results = [{'content': "execise!"}]

    llm = NationalServiceRAGLLM(system_config)
    async for result, text in llm.astructured_chat(
            formatted_prompt,
            search_results,
            user_input,
            state.user_persona,
            state.user_bio,
            [],
    ):
        print(result)


def main():
    # demo_invoke()

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_astream())
    loop.close()


if __name__ == "__main__":
    main()
