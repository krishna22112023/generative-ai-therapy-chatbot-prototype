from typing import Optional, TypedDict, Annotated
from pydantic import BaseModel
from core.utils import (
    extract_survey_response,
    save_survey_responses,
    create_dynamic_survey_answer,
    score_and_interpret_latest_survey,
    save_scored_survey,
)
from openai import OpenAI
from langchain.schema import HumanMessage
import instructor
import json
import logging
from conf.config import cfg

from core.utils import get_chat_model, format_chat_history, dict2xml


logger = logging.getLogger(__name__)


class SurveyAnswer(BaseModel):
    Question1: str
    Answer1: str
    Question2: str
    Answer2: str

class Result(TypedDict):
    state: Annotated[str, ..., "The routing state based on user query in <state> tag."]
    thinking: Annotated[str, ..., "The thinking based on the user query in <thinking> tag."]
    response: Annotated[str, ..., "The response based on the user query in <response> tag."]


class SurveyLLM:
    def __init__(self, args):
        self.chat_model = get_chat_model()
        self.file_path = args.get("survey_answers_file_path")
        self.survey_name = args.get("survey_name")

    def _format_survey_history(self, chat_history) -> str:

        # Look for the last assistant message containing the survey summary
        for message in reversed(chat_history):
            if (
                message["role"] == "assistant"
                and "Survey Summary" in message["content"]
            ):
                logger.info(
                    f"Found survey summary in message: {message['content'][:100]}..."
                )  # Print first 100 chars
                return message["content"]

        # If no summary found, construct one from user responses
        summary = "Survey Summary:\n"
        for message in chat_history:
            if message["role"] == "user" and message["content"] in ["a", "b", "c", "d"]:
                summary += f"User response: {message['content']}\n"

        if summary == "Survey Summary:\n":
            logger.error("No survey responses found in chat history")
            return "No survey responses recorded."
        else:
            logger.info(f"Constructed summary: {summary}")
            return summary

    def structured_chat(self, formatted_survey_prompt, user_input: str,
                        survey_qa: str, chat_history: Optional[list] = None):
        if chat_history is None:
            chat_history = []

        formatted_history = format_chat_history(chat_history)
        formatted_history.append(HumanMessage(content=user_input))

        chain = formatted_survey_prompt | self.chat_model.with_structured_output(Result)
        inputs = {
            'messages': formatted_history,
            'survey_qa': survey_qa
        }

        result = chain.invoke(inputs)
        text = dict2xml(result)
        return result, text

    async def astructured_chat(self, formatted_survey_prompt, user_input: str,
                        survey_qa: str, chat_history: Optional[list] = None):
        if chat_history is None:
            chat_history = []

        formatted_history = format_chat_history(chat_history)
        formatted_history.append(HumanMessage(content=user_input))

        chain = formatted_survey_prompt | self.chat_model.with_structured_output(Result)
        inputs = {
            'messages': formatted_history,
            'survey_qa': survey_qa
        }

        last_chunk = {}
        async for chunk in chain.astream(inputs):
            last_chunk = chunk
            yield chunk, ""

        result = last_chunk
        text = dict2xml(result)
        yield result, text

    def save_structured_survey_chat(self, username, chat_history, survey_name):
        # Load the survey structure
        # with open(f"./data/raw/survey/survey_{survey_name}_questions.json", "r") as f:
        #     survey_structure = json.load(f)
        print('chat history', type(chat_history))
        print(chat_history)

        survey_path = getattr(cfg.survey, f"survey_{survey_name}_questions")
        with open(survey_path, "r") as f:
            survey_structure = json.load(f)

        # Create the dynamic SurveyAnswer class
        DynamicSurveyAnswer = create_dynamic_survey_answer(survey_structure)

        # Extract survey responses data from conversation just before end state
        survey_summary_history = self._format_survey_history(chat_history)
        client = instructor.from_openai(OpenAI())

        try:
            survey_answer = client.chat.completions.create(
                model="gpt-4o",
                response_model=DynamicSurveyAnswer,
                messages=[
                    {
                        "role": "user",
                        "content": {"type": "text", "text": survey_summary_history},
                    }
                ],
            )
        except Exception as e:
            survey_answer = client.chat.completions.create(
                model="gpt-4o",
                response_model=DynamicSurveyAnswer,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": survey_summary_history}],
                    }
                ],
            )

        save_survey_responses(survey_answer, self.file_path, username, survey_name)

        # Score and interpret the survey
        result = score_and_interpret_latest_survey(
            username, survey_name, self.file_path
        )
        # Log the result
        if result["error"]:
            logger.error(
                f"Error scoring survey for user {username}, survey {survey_name}: {result['error']}"
            )
        else:
            logger.info(
                f"Survey scored for user {username}, survey {survey_name}. Score available: {result['score_available']}"
            )

            # Save the scored survey result only if score is available
            if result["score_available"]:
                saved = save_scored_survey(username, survey_name, result)
                if saved:
                    logger.info(
                        f"Scored survey saved for user {username}, survey {survey_name}, date {result['survey_date']}"
                    )
                else:
                    logger.error(
                        f"Failed to save scored survey for user {username}, survey {survey_name}, date {result['survey_date']}"
                    )
            else:
                logger.info(
                    f"Score not available for user {username}, survey {survey_name}. Scored result not saved."
                )

        return result


def demo_invoke():
    from conf.config import system_config, cfg
    from core.state import SurveyState
    from core.utils import check_surveys_due

    username = 'benjamin.davis'
    survey_due_results = check_surveys_due(
        username,
        system_config["days_thresholds"],
        system_config["survey_answers_file_path"],
    )
    state = SurveyState("duringNS", cfg.prompts.survey_system_during_NS, survey_due_results)
    formatted_prompt = state.enter()
    user_input = 'Hello'

    llm = SurveyLLM(system_config)

    reply, text = llm.structured_chat(formatted_prompt, user_input, state.survey_qa, [])
    print(reply)


async def demo_astream():
    from conf.config import system_config, cfg
    from core.state import SurveyState
    from core.utils import check_surveys_due

    username = 'benjamin.davis'
    survey_due_results = check_surveys_due(
        username,
        system_config["days_thresholds"],
        system_config["survey_answers_file_path"],
    )
    state = SurveyState("duringNS", cfg.prompts.survey_system_during_NS, survey_due_results)
    formatted_prompt = state.enter()
    user_input = 'Hello'

    llm = SurveyLLM(system_config)

    async for reply, text in llm.astructured_chat(formatted_prompt, user_input, state.survey_qa, []):
        print(reply)


def main():
    # demo_invoke()

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_astream())
    loop.close()


if __name__ == "__main__":
    main()