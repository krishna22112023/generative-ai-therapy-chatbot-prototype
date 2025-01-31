import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from conf.config import system_config
from core.response_llm import response_validation
from core.logic import NationalServiceLogic, LLMTransitionLogic
from core.state import NationalServiceApp
from general.utils import Redis

logger = logging.getLogger(__name__)


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

    # Generate response
    async for chunk in llm_router.agen_response(
            full_user_query, state["chat_history"], state["args"]):
        yield chunk


async def demo_async():
    user_name = "benjamin.davis"
    session_id = 'S0001'
    query = "What is national service in Singapore?"

    Redis.clear(session_id)
    async for chunk in achat(user_name, session_id, query):
        print(chunk)


def main():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_async())
    loop.close()


if __name__ == "__main__":
    main()
