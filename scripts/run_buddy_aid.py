import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datetime import datetime
from core.response_llm import response_validation
from core.logic import NationalServiceLogic, LLMTransitionLogic
from core.state import NationalServiceApp
import logging
from conf.config import cfg, system_config


logger = logging.getLogger(__name__)


def run():
    # arguments
    username = "benjamin.davis"
    args = system_config
    # checking NS state is already set. If already, skip the NS state logic!
    ns_logic = NationalServiceLogic(username)
    if ns_logic.check_last_ns_state() is None:
        ns_logic.get_current_ns_stage()
        ns_state = ns_logic.check_last_ns_state()
    else:
        ns_state = ns_logic.check_last_ns_state()
    #
    ns_app = NationalServiceApp(username, args)
    llm_states = getattr(ns_app, ns_state)()

    llm_router = LLMTransitionLogic(
        ns_state,
        username,
        llm_states,
        args["days_thresholds"],
        args["survey_answers_file_path"],
    )

    chat_history = []  # Initialize empty chat history
    question_counter = 0  # Initialize question counter

    state = {
        "username": username,
        "args": args,
        "ns_state": ns_state,
        "llm_router": llm_router,
        "chat_history": [],
    }

    # Continuous interaction loop
    while True:
        user_query = input("\n User: ")  # Get input from user
        if user_query.lower() in ["exit", "quit", "end"]:  # Check for exit condition
            logger.info("Session ended. Thank you!")
            break
        # Increment question counter and update system message
        question_counter += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        day_of_week = datetime.now().strftime("%A")
        # [MP] eventually, this event will be passed to the Event Manager
        # logger.info(extract_save_event(response))

        # Create preamble with additional information
        preamble = f"[Additional Information: Current Date Time is {current_time}, Day of the Week: {day_of_week}] \n "

        # Add user message to chat history with preamble
        full_user_query = preamble + user_query
        chat_history.append({"role": "user", "content": full_user_query})
        print("******************************* \n")
        # Generate a response based on the user query and chat history
        response, llm_state = llm_router.gen_response(
            full_user_query, state["chat_history"], args
        )

        # print sources if RAG state is enabled
        if (llm_state == "continue_counsellor") | (llm_state == "continue_RAG_NS"):
            urls = response[1]
            if isinstance(urls, list):
                print("\n\n---Sources---")
                for index, item in enumerate(urls, start=1):
                    logger.info(f"{index}. {item}")
                print("-------\n")
                response = response[0]
        elif (llm_state == "continue_survey") | (llm_state == "end_survey"):
            response = response[0]

        validated_response = response_validation(
            username, response, full_user_query, chat_history
        )

        # temporary fix
        if isinstance(validated_response, tuple):
            validated_response = validated_response[0]
        # Update chat history
        if validated_response:
            state["chat_history"].append(
                {"role": "assistant", "content": validated_response}
            )
            # logger.info(f"\nAssistant: {validated_response}")
        print("******************************* \n")


if __name__ == "__main__":
    run()
