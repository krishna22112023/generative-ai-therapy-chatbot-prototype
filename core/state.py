from datetime import datetime
from core.utils import (
    load_bio,
    load_facts,
    read_text_file,
    load_survey_name_due,
    score_and_interpret_latest_survey,
    save_scored_survey,
    check_surveys_due,
)
from llms.llm_empathetic.chat_completion import EmpatheticLLM
from llms.llm_empathetic.prompt_format import create_empathetic_prompt
from llms.llm_counsellor.chat_completion import CounsellorLLM
from llms.llm_counsellor.prompt_format import create_counsellor_prompt
from llms.llm_survey.chat_completion import SurveyLLM
from llms.llm_survey.prompt_format import create_survey_prompt
from llms.llm_rag.chat_completion import NationalServiceRAGLLM
from llms.llm_rag.prompt_format import create_ns_rag_prompt
from llms.llm_rag.retriever import retriever_ns_rag
from llms.llm_counsellor.retriever import retriever_counsellor_rag
from core.state_template import ConvoState, AppState
import os
import logging
import json
from conf.config import cfg

logger = logging.getLogger(__name__)


class EmpatheticState(ConvoState):
    def __init__(self, name, template_path, username):
        super().__init__(name)

        self.template_path = template_path
        self.user_persona = load_facts(cfg.user.facts, username)
        self.user_bio = load_bio(cfg.user.bio, username)

    def enter(self):
        logger.info(f"\033[32mEntering {self.name} Empathetic State\033[0m")
        # load and format prompt
        template = read_text_file(self.template_path)
        prompt_args = ["user_persona_prompt", "user_bio"]
        final_template = create_empathetic_prompt(template, prompt_args)
        return final_template

    def execute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[32mExecuting {self.name} Empathetic State\033[0m")
        # chat completion
        LLM = EmpatheticLLM(args)
        structured_response, raw_response = LLM.chat(
            formatted_prompt,
            user_input,
            self.user_persona,
            self.user_bio,
            chat_history,
        )
        return structured_response, raw_response

    async def aexecute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[32mExecuting {self.name} Empathetic State\033[0m")
        llm = EmpatheticLLM(args)
        async for result, text in llm.achat(
                formatted_prompt,
                user_input,
                self.user_persona,
                self.user_bio,
                chat_history,
        ):
            yield result, text

    def end(self):
        logger.info(f"\033[34mEnding {self.name} Empathetic State\033[0m")


class CounsellorState(ConvoState):
    def __init__(self, name, system_template_path, human_template_path, username):
        super().__init__(name)

        self.system_template_path = system_template_path
        self.human_template_path = human_template_path
        self.user_persona = load_facts(cfg.user.facts, username)
        self.user_bio = load_bio(cfg.user.bio, username)
        self.router_prompt = cfg.prompts.router
        self.username = username

    def enter(self):
        logger.info(f"\033[34mEntering {self.name} RAG_Counsellor State\033[0m")
        # load and format prompt
        system_template = read_text_file(self.system_template_path)
        human_template = read_text_file(self.human_template_path)
        system_prompt_args = ["user_persona_prompt", "user_bio", "context", "question"]
        human_prompt_args = ["context", "question"]
        final_template = create_counsellor_prompt(
            system_template, system_prompt_args, human_template, human_prompt_args
        )
        return final_template

    def execute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[34mExecuting {self.name} RAG_Counsellor State\033[0m")

        yield {'log': "Fetching documents now...\n"}, "", []

        search_results = retriever_counsellor_rag(args, user_input)
        urls = [result["metadata"]["url"] for result in search_results]

        yield {'log': f"Fetched {len(search_results)} documents\n"}, "", []

        # chat completion
        llm = CounsellorLLM(args)
        result, text = llm.structured_chat(
            formatted_prompt,
            search_results,
            user_input,
            self.user_persona,
            self.user_bio,
            chat_history,
        )
        return result, text, urls

    async def aexecute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[34mExecuting {self.name} RAG_Counsellor State\033[0m")

        search_results = retriever_counsellor_rag(args, user_input)
        urls = [result["metadata"]["url"] for result in search_results]

        LLM = CounsellorLLM(args)
        async for result, text in LLM.astructured_chat(
            formatted_prompt,
            search_results,
            user_input,
            self.user_persona,
            self.user_bio,
            chat_history,
        ):
            yield result, text, urls

    def end(self):
        logger.info(f"\033[34mEnding {self.name} RAG_Counsellor State\033[0m")


class SurveyState(ConvoState):
    def __init__(self, name, template_path, survey_due_results):
        super().__init__(name)
        self.template_path = template_path
        self.survey_name = load_survey_name_due(survey_due_results)
        self.survey_qa = self._load_survey()
        logging.info(f"Initialized SurveyState with survey: {self.survey_name}")

    def update_survey_info(self, survey_name):
        self.survey_name = survey_name
        self.survey_qa = self._load_survey()

    def _load_survey(self):
        if not self.survey_name:
            logging.warning("No survey name provided")
            return None

        survey_path = getattr(cfg.survey, f"survey_{self.survey_name}_questions")

        if os.path.exists(survey_path):
            with open(survey_path, "r") as file:
                survey_json = json.load(file)

            formatted_survey = "Preamble:\n"
            formatted_survey += f"{survey_json['preamble']}\n\n"
            formatted_survey += "Questions and answers:\n"

            for question_set in survey_json["qa"]:
                for q_num, q_text in question_set.items():
                    formatted_survey += f"Question {q_num}: {q_text}\n"

            return formatted_survey
        else:
            logging.error(f"Survey file not found: {survey_path}")
            return None

    def enter(self):
        logger.info(f"\033[36mEntering {self.name} Survey State\033[0m")
        # load and format prompt
        template = read_text_file(self.template_path)
        prompt_args = ["survey_qa"]
        final_template = create_survey_prompt(template, prompt_args)
        return final_template

    def execute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[36mExecuting {self.name} Survey State\033[0m")

        # Ensure survey_qa is loaded
        if self.survey_qa is None:
            logging.warning("survey_qa is None. Attempting to reload.")
            self.survey_qa = self._load_survey()
            if self.survey_qa is None:
                logging.error("Failed to load survey_qa. Cannot execute survey.")
                return None, "Error: Unable to load survey questions."

        # chat completion
        llm = SurveyLLM(args)
        result, text = llm.structured_chat(formatted_prompt, user_input, self.survey_qa, chat_history)
        return result, text

    async def aexecute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[36mExecuting {self.name} Survey State\033[0m")

        if self.survey_qa is None:
            logging.warning("survey_qa is None. Attempting to reload.")
            self.survey_qa = self._load_survey()
            if self.survey_qa is None:
                logging.error("Failed to load survey_qa. Cannot execute survey.")
                raise ValueError("Error: Unable to load survey questions.")

        # chat completion
        llm = SurveyLLM(args)
        async for result, text in llm.astructured_chat(formatted_prompt, user_input, self.survey_qa, chat_history):
            yield result, text

    def end(self, args, username, chat_history):
        logger.info(f"\n\033[36mEnding {self.name} Survey State\033[0m")
        LLM = SurveyLLM(args)
        LLM.save_structured_survey_chat(username, chat_history, self.survey_name)
        logger.info(f"\n\033[33mSaved survey responses\033[0m")

        # Score and interpret the survey
        result = score_and_interpret_latest_survey(
            username, self.survey_name, args.get("survey_answers_file_path")
        )
        feedback_message = None

        if result["error"]:
            logger.error(
                f"Error scoring survey for user {username}, survey {self.survey_name}: {result['error']}"
            )
        else:
            logger.info(
                f"Survey scored for user {username}, survey {self.survey_name}. Score available: {result['score_available']}"
            )

            # Save the scored survey result only if score is available
            if result["score_available"]:
                saved = save_scored_survey(username, self.survey_name, result)
                if saved:

                    # Generate feedback message
                    feedback_message = {'state':'end_survey',
                                        'response':f"\nOn your recent {self.survey_name} survey you scored {result['score']}\nThis is how you can interpret the score:\n{result['interpretation']}"
                    }
                else:
                    logger.error(
                        f"Failed to save scored survey for user {username}, survey {self.survey_name}, date {result['survey_date']}. Check previous logs for details."
                    )
            else:
                logger.info(
                    f"Score not available for user {username}, survey {self.survey_name}. Scored result not saved."
                )

        return feedback_message


class NationalServiceRAGState(ConvoState):
    def __init__(self, name, system_template_path, human_template_path, username):
        super().__init__(name)
        self.system_template_path = system_template_path
        self.human_template_path = human_template_path
        self.user_persona = load_facts(cfg.user.facts, username)
        self.user_bio = load_bio(cfg.user.bio, username)

    def enter(self):
        logger.info(f"\033[34mEntering {self.name} RAG_NS State\033[0m")
        # load and format prompt
        system_template = read_text_file(self.system_template_path)
        human_template = read_text_file(self.human_template_path)
        system_prompt_args = ["user_persona_prompt", "user_bio", "context", "question"]
        human_prompt_args = ["context", "question"]
        final_template = create_ns_rag_prompt(
            system_template, system_prompt_args, human_template, human_prompt_args
        )
        return final_template

    def execute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[34mExecuting {self.name} RAG_NS State\033[0m")

        # retrieving search results and metadata urls
        search_results = retriever_ns_rag(args, user_input)
        urls = [result["metadata"]["url"] for result in search_results]

        # chat completion
        llm = NationalServiceRAGLLM(args)
        result, text = llm.structured_chat(
            formatted_prompt,
            search_results,
            user_input,
            self.user_persona,
            self.user_bio,
            chat_history,
        )
        return result, text, urls

    async def aexecute(self, formatted_prompt, user_input, chat_history, args):
        logger.info(f"\033[34mExecuting {self.name} RAG_NS State\033[0m")

        yield {'log': "Fetching documents now...\n"}, "", []

        search_results = retriever_ns_rag(args, user_input)
        urls = [result["metadata"]["url"] for result in search_results]

        yield {'log': f"Fetched {len(search_results)} documents\n"}, "", []

        # chat completion
        llm = NationalServiceRAGLLM(args)
        async for result, text in llm.astructured_chat(
            formatted_prompt,
            search_results,
            user_input,
            self.user_persona,
            self.user_bio,
            chat_history,
        ):
            yield result, text, urls

    def end(self):
        logger.info(f"\033[34mEnding {self.name} RAG_NS State\033[0m")


class NationalServiceApp(AppState):
    def __init__(self, username, args):
        print(f"Config received in NationalServiceApp.__init__ (state.py): {args}")
        super().__init__(username)
        self.username = username
        self.args = args
        self.survey_due_results = check_surveys_due(
            username, args["days_thresholds"], args["survey_answers_file_path"]
        )

    def preNS(self):
        empathetic_template = cfg.prompts.empathetic_pre_NS
        empathetic_state = EmpatheticState("preNS", empathetic_template, self.username)

        Counsellor_NS_template = cfg.prompts.counsellor_system_pre_NS
        Counsellor_NS_human_template = cfg.prompts.counsellor_human_NS
        counsellor_rag_state = CounsellorState(
            "preNS", Counsellor_NS_template, Counsellor_NS_human_template, self.username
        )

        RAG_NS_template = cfg.prompts.rag_ns_system_pre_NS
        RAG_NS_human_template = cfg.prompts.rag_ns_human_NS
        ns_rag_state = NationalServiceRAGState(
            "preNS", RAG_NS_template, RAG_NS_human_template, self.username
        )

        return (empathetic_state, counsellor_rag_state, ns_rag_state)

    def duringNS(self):
        empathetic_template = cfg.prompts.empathetic_during_NS
        empathetic_state = EmpatheticState(
            "duringNS", empathetic_template, self.username
        )

        Counsellor_NS_template = cfg.prompts.counsellor_system_during_NS
        Counsellor_NS_human_template = cfg.prompts.counsellor_human_NS
        counsellor_rag_state = CounsellorState(
            "duringNS",
            Counsellor_NS_template,
            Counsellor_NS_human_template,
            self.username,
        )

        survey_due_results = check_surveys_due(
            self.username,
            self.args["days_thresholds"],
            self.args["survey_answers_file_path"],
        )
        survey_template = cfg.prompts.survey_system_during_NS
        survey_state = SurveyState("duringNS", survey_template, survey_due_results)

        RAG_NS_template = cfg.prompts.rag_ns_system_during_NS
        RAG_NS_human_template = cfg.prompts.rag_ns_human_NS
        ns_rag_state = NationalServiceRAGState(
            "duringNS", RAG_NS_template, RAG_NS_human_template, self.username
        )

        return (empathetic_state, counsellor_rag_state, survey_state, ns_rag_state)

    def postNS(self):
        empathetic_template = cfg.prompts.empathetic_post_NS
        empathetic_state = EmpatheticState("postNS", empathetic_template, self.username)

        Counsellor_NS_template = cfg.prompts.counsellor_system_post_NS
        Counsellor_NS_human_template = cfg.prompts.counsellor_human_NS
        counsellor_rag_state = CounsellorState(
            "postNS",
            Counsellor_NS_template,
            Counsellor_NS_human_template,
            self.username,
        )

        RAG_NS_template = cfg.prompts.rag_ns_system_post_NS
        RAG_NS_human_template = cfg.prompts.rag_ns_human_NS
        ns_rag_state = NationalServiceRAGState(
            "postNS", RAG_NS_template, RAG_NS_human_template, self.username
        )

        return (empathetic_state, counsellor_rag_state, ns_rag_state)


async def demo_aempathetic():
    from conf.config import system_config, cfg
    state = EmpatheticState("duringNS", cfg.prompts.empathetic_during_NS, 'benjamin.davis')
    formatted_prompt = state.enter()
    user_input = 'Hello'
    async for result, text in state.aexecute(formatted_prompt, user_input, [], system_config):
        print(result)


async def demo_acounsellor():
    from conf.config import system_config, cfg

    state = CounsellorState(
        "duringNS",
        cfg.prompts.counsellor_system_during_NS,
        cfg.prompts.counsellor_human_NS,
        'benjamin.davis'
    )
    formatted_prompt = state.enter()
    user_input = "give me some tips on building muscle"

    async for result, text, urls in state.aexecute(formatted_prompt, user_input, [], system_config):
        print(result)


async def demo_asurvey():
    from conf.config import system_config, cfg

    username = 'benjamin.davis'
    survey_due_results = check_surveys_due(
        username,
        system_config["days_thresholds"],
        system_config["survey_answers_file_path"],
    )
    state = SurveyState("duringNS", cfg.prompts.survey_system_during_NS, survey_due_results)
    formatted_prompt = state.enter()
    user_input = 'Hello'

    async for result, text in state.aexecute(formatted_prompt, user_input, [], system_config):
        print(result)


async def demo_arag_ns():
    from conf.config import system_config, cfg

    state = NationalServiceRAGState("duringNS", cfg.prompts.rag_ns_system_during_NS,
                                    cfg.prompts.rag_ns_human_NS, 'benjamin.davis')
    formatted_prompt = state.enter()
    user_input = "What is national service in Singapore"

    async for result, text, urls in state.aexecute(formatted_prompt, user_input, [], system_config):
        print(result)

def main():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_arag_ns())
    loop.close()



if __name__ == '__main__':
    main()
