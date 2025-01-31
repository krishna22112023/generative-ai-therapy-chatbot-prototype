import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import re
import os
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
from conf.config import system_config
from core.response_llm import response_validation
from core.logic import NationalServiceLogic, LLMTransitionLogic
from core.state import NationalServiceApp
from general.utils import Redis
from core.utils import download_gdrive_image

logger = logging.getLogger(__name__)


def chat(username: str, session_id: str, query: str):
    ns_logic = NationalServiceLogic(username)

    if ns_logic.check_last_ns_state() is None:
        ns_logic.get_current_ns_stage()
    ns_state = ns_logic.check_last_ns_state()

    state = Redis.load(session_id)
    if state is None:
        state = {
            "username": username,
            "args": system_config,
            "ns_state": ns_state,
            "chat_history": [],
            "turn_count": 0,
            "llm_state": None
        }
    else:
        state = Redis.load(session_id)
        state['args'] = system_config
        state["ns_state"] = ns_state

    ns_app = NationalServiceApp(username, system_config)
    llm_states = getattr(ns_app, ns_state)()

    llm_router = LLMTransitionLogic(
        ns_state,
        username,
        llm_states,
        state,
        system_config["days_thresholds"],
        system_config["survey_answers_file_path"],
    )

    # Process user input
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day_of_week = datetime.now().strftime("%A")
    preamble = f"[Additional Information: Current Date Time is {current_time}, Day of the Week: {day_of_week}] \n "
    full_user_query = preamble + query
    state["chat_history"].append({"role": "user", "content": full_user_query})

    # Generate response
    response, llm_state = llm_router.gen_response(
        full_user_query, state["chat_history"], state["args"]
    )
    logger.info(f"Response from buddyAID {response}")

    # Process response
    if (llm_state == "continue_counsellor") | (llm_state == "continue_RAG_NS"):
        urls = response[1]
        if isinstance(urls, list):
            sources = (
                    "<sources> \n Sources \n"
                    + "\n".join([f"{i + 1}. {url}" for i, url in enumerate(urls)])
                    + "</sources>"
            )
            response = response[0] + sources
    elif (llm_state == "continue_survey") | (llm_state == "end_survey"):
        response = response[0]

    validated_response = response_validation(
        state["username"], response, full_user_query, state["chat_history"]
    )

    if isinstance(validated_response, tuple):
        validated_response = validated_response[0]

    state["chat_history"].append({"role": "assistant", "content": validated_response})
    state["turn_count"] = state.get("turn_count", 1) + 1
    state["llm_state"] = llm_state

    Redis.save(session_id, state)

    return json.dumps({"response": validated_response, "state": json.dumps(state)})


async def achat(username: str, session_id: str, query: str):
    ns_logic = NationalServiceLogic(username)

    if ns_logic.check_last_ns_state() is None:
        ns_logic.get_current_ns_stage()
    ns_state = ns_logic.check_last_ns_state()

    state = Redis.load(session_id)

    if state is None:
        state = {
            "username": username,
            "args": system_config,
            "ns_state": ns_state,
            "chat_history": [],
            "turn_count": 0,
            "llm_state": None
        }
    else:
        state = Redis.load(session_id)
        state['args'] = system_config
        state["ns_state"] = ns_state

    ns_app = NationalServiceApp(username, system_config)
    llm_states = getattr(ns_app, ns_state)()

    llm_router = LLMTransitionLogic(
        ns_state,
        username,
        llm_states,
        state,
        system_config["days_thresholds"],
        system_config["survey_answers_file_path"],
    )

    # Process user input
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day_of_week = datetime.now().strftime("%A")
    preamble = f"[Additional Information: Current Date Time is {current_time}, Day of the Week: {day_of_week}] \n "
    full_user_query = preamble + query
    state["chat_history"].append({"role": "user", "content": full_user_query})

    response = {}
    last_response = None
    urls = []
    log_len, thinking_len, response_len = 0, 0, 0
    async for response in llm_router.agen_response(full_user_query, state["chat_history"], state["args"]):
        logger.info(f"sturctured response streamed out {response}")
        if isinstance(response, tuple) and len(response) == 2:
            response, urls = response
        if 'log' in response and 'thinking' not in response and 'response' not in response:
            text = response['log']
            yield text
        '''elif 'thinking' in response and 'response' not in response:
            last_response = response
            if len(response['thinking']) > thinking_len:
                text = response['thinking'][thinking_len:]
                if thinking_len == 0:
                    yield 'Thinking: \n'+text
                else:
                    yield text
                thinking_len = len(response['thinking'])'''
        if 'response' in response:
            last_response = response
            if len(response['response']) > response_len:
                text = response['response'][response_len:]
                if response_len == 0 and 'thinking' in response:
                    yield '\n'+text
                else:
                    yield text
                response_len = len(response['response'])
        else:
            pass
    last_state = last_response["state"]
    if last_state is not None and ("counsel" in last_state or "RAG_NS" in last_state):
        if urls:
            sources = ("\n Sources \n" + "\n".join([f"{i + 1}. {url}" for i, url in enumerate(list(set(urls)))]))
            yield sources
        os.makedirs(os.path.join(base_path,"open_webui/build/images"),exist_ok=True)
        for url in list(set(urls)):
            if "drive.google.com" in url:
                file_name = download_gdrive_image(url,os.path.join(base_path,"open_webui/build/images"))
                yield f"\n![Test](/images/{file_name})\n"
    logger.info(f"Response from buddyAID {response}")

    llm_state = last_response['state']
    llm_state = re.sub(r'[<>]', '', llm_state)

    validated_response = response_validation(
        state["username"], last_response['response'], full_user_query, state["chat_history"]
    )

    if isinstance(validated_response, tuple):
        validated_response = validated_response[0]

    state["chat_history"].append({"role": "assistant", "content": validated_response})
    state["turn_count"] = state.get("turn_count", 1) + 1
    state["llm_state"] = llm_state
    print('state', llm_state)

    Redis.save(session_id, state)


async def demo_async():
    user_name = "benjamin.davis"
    session_id = 'S0001'
    query = "hello"

    Redis.clear(session_id)
    async for result in achat(user_name, session_id, query):
        print(result)


def main():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_async())
    loop.close()


if __name__ == "__main__":
    main()
