from datetime import datetime
from core.utils import (
    load_bio,
    check_gad7_due_status,
    check_surveys_due,
    get_max_days_surveys_due,
    load_survey_name_due,
    read_text_file,
)
from semantic_layer import LLM_transition, LLM_router
from core.response_llm import ResponseHandler

import re
import logging
from conf.config import cfg

logger = logging.getLogger(__name__)


class NationalServiceLogic:
    """
    Manages the logic to determine which National Service stage to activate based on the user's service dates.
    """

    def __init__(self, username):
        """
        Initialize the logic with the username to load specific user data.

        Args:
        username (str): Username of the national service member.
        """
        self.username = username
        self.user_bio = load_bio(cfg.user.bio, username)
        self.current_ns_state = None

    def get_current_ns_stage(self):
        """
        Determines the current NS state based on today's date and the user's NS service dates.

        Returns:
        str: The current NS stage ('PreNS', 'DuringNS', 'PostNS')
        """
        today = datetime.now().date()

        pattern_ns_start = r"ns_start:\s*(\d{4}-\d{2}-\d{2})"
        ns_start_match = re.search(pattern_ns_start, self.user_bio)
        ns_start = ns_start_match.group(1) if ns_start_match else None

        pattern_ns_stop = r"ns_stop:\s*(\d{4}-\d{2}-\d{2})"
        ns_stop_match = re.search(pattern_ns_stop, self.user_bio)
        ns_stop = ns_stop_match.group(1) if ns_stop_match else None

        if ns_start and ns_stop:
            ns_start_date = datetime.strptime(ns_start, "%Y-%m-%d").date()
            ns_stop_date = datetime.strptime(ns_stop, "%Y-%m-%d").date()

            if today < ns_start_date:
                self.current_ns_state = "preNS"
            elif ns_start_date <= today <= ns_stop_date:
                self.current_ns_state = "duringNS"
            elif today > ns_stop_date:
                self.current_ns_state = "postNS"
        else:
            self.current_ns_state = (
                "duringNS"  # Default state if dates are not specified
            )

    def check_last_ns_state(self):
        return self.current_ns_state


class LLMTransitionLogic:
    def __init__(
        self,
        ns_state,
        username,
        llm_states,
        state,
        days_thresholds,
        survey_answers_file_path,
    ):
        self.username = username
        self.router = LLM_transition.LLMTransition().get_routes()
        self.handler = ResponseHandler(ns_state, username, llm_states)
        self.router_prompt = cfg.prompts.router
        self.days_thresholds = days_thresholds
        self.survey_answers_file_path = survey_answers_file_path
        self.current_state = state["llm_state"]
        self.state = {
            "surveys": self.update_surveys_due(),
        }
        self.state["survey_name"] = self.get_current_survey_name()

    def update_surveys_due(self):
        return check_surveys_due(
            self.username, self.days_thresholds, self.survey_answers_file_path
        )

    def get_current_survey_name(self):
        return load_survey_name_due(self.state["surveys"])

    def refresh_survey_info(self):
        self.state["surveys"] = self.update_surveys_due()
        self.state["survey_name"] = self.get_current_survey_name()

    def gen_response(self, user_input, chat_history, args):
        print(f"Current llm state {self.current_state}")
        # Update the surveys due status before generating a response
        self.refresh_survey_info()

        survey_due_days = get_max_days_surveys_due(self.state["surveys"])

        if (survey_due_days < 5) & (self.handler.check_turn_count() > 3):
            logger.info(
                f"\033[31mSurvey is due in {survey_due_days}. User is reminded to take survey.\033[0m"
            )
            self.handler.update_survey_state(self.state["survey_name"])
            response = self.handler.generate_survey_response(
                user_input, chat_history, args, self.username, self.router_prompt
            )
        else:
            if self.current_state == "continue_counsellor":
                response = self.handler.generate_counsellor_response(
                    user_input, chat_history, args, self.router_prompt
                )
            elif self.current_state == "continue_survey":
                self.handler.update_survey_state(self.state["survey_name"])
                response = self.handler.generate_survey_response(
                    user_input, chat_history, args, self.username, self.router_prompt
                )
            elif self.current_state == "continue_RAG_NS":
                response = self.handler.generate_rag_ns_response(
                    user_input, chat_history, args, self.router_prompt
                )
            elif self.current_state == "continue_empathetic":
                response = self.handler.generate_empathetic_response(
                    user_input, chat_history, args, self.router_prompt
                )
                
            else:
                # check state determined by llm router
                template = read_text_file(self.router_prompt)
                LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(
                    template
                )
                Router_llm = LLM_router.chat_completion.RouterLLM(args)
                llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
                print(f"LLM state selected by planner: {llm_route}")
                if "counselor" in llm_route.get("state"):
                    response = self.handler.generate_counsellor_response(
                        user_input, chat_history, args, self.router_prompt
                    )
                    self.current_state = "continue_counsellor"
                elif "survey" in llm_route.get("state"):
                    response = self.handler.generate_survey_response(
                        user_input,
                        chat_history,
                        args,
                        self.username,
                        self.router_prompt,
                    )
                    self.current_state = "continue_survey"
                elif "RAG_NS" in llm_route.get("state"):
                    response = self.handler.generate_rag_ns_response(
                        user_input, chat_history, args, self.router_prompt
                    )
                    self.current_state = "continue_RAG_NS"
                elif "empathetic" in llm_route.get("state"):
                    response = self.handler.generate_empathetic_response(
                        user_input, chat_history, args, self.router_prompt
                    )
                    self.current_state = "continue_empathetic"
                # Fallback router : check state using semantic router if undetermined by llm router
                else:
                    logger.info(
                        f"LLM state undecided by planner. Falling back to semantic router"
                    )
                    route = self.router(user_input)
                    if route.name == "Counsellor":
                        response = self.handler.generate_counsellor_response(
                            user_input, chat_history, args, self.router_prompt
                        )
                        self.current_state = "continue_counsellor"
                    elif route.name == "Survey":
                        response = self.handler.generate_survey_response(
                            user_input,
                            chat_history,
                            args,
                            self.username,
                            self.router_prompt,
                        )
                        self.current_state = "continue_survey"
                    elif route.name == "RAG_NS":
                        response = self.handler.generate_rag_ns_response(
                            user_input, chat_history, args, self.router_prompt
                        )
                        self.current_state = "continue_RAG_NS"
                    else:
                        response = self.handler.generate_empathetic_response(
                            user_input, chat_history, args, self.router_prompt
                        )
                        self.current_state = "continue_empathetic"

        return response, self.current_state

    async def agen_response(self, user_input, chat_history, args):
        print(f"Current llm state {self.current_state}")
        # Update the surveys due status before generating a response
        self.refresh_survey_info()

        survey_due_days = get_max_days_surveys_due(self.state["surveys"])

        if (survey_due_days < 5) & (self.handler.check_turn_count() > 3):
            logger.info(
                f"\033[31mSurvey is due in {survey_due_days}. User is reminded to take survey.\033[0m"
            )
            self.handler.update_survey_state(self.state["survey_name"])
            async for chunk in self.handler.agenerate_survey_response(
                user_input, chat_history, args, self.username, self.router_prompt
            ):
                yield chunk
        else:
            if self.current_state == "continue_counsellor":
                async for chunk in self.handler.agenerate_counsellor_response(
                        user_input, chat_history, args, self.router_prompt):
                    yield chunk
            elif self.current_state == "continue_survey":
                self.handler.update_survey_state(self.state["survey_name"])
                async for chunk in self.handler.agenerate_survey_response(
                        user_input, chat_history, args, self.username, self.router_prompt):
                    yield chunk
            elif self.current_state == "continue_RAG_NS":
                async for chunk in self.handler.agenerate_rag_ns_response(
                        user_input, chat_history, args, self.router_prompt):
                    yield chunk
            elif self.current_state == "continue_empathetic":
                async for chunk in self.handler.agenerate_empathetic_response(
                        user_input, chat_history, args, self.router_prompt):
                    yield chunk
            else:
                yield {'log': 'Planning...\n'}
                # check state determined by llm router
                template = read_text_file(self.router_prompt)
                LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(
                    template
                )
                Router_llm = LLM_router.chat_completion.RouterLLM(args)
                llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
                state = llm_route.get("state")
                yield {'log': f'Thinking...\n'}
                print(f"LLM state selected by planner: {state}")

                if "counselor" in state:
                    async for chunk in self.handler.agenerate_counsellor_response(
                            user_input, chat_history, args, self.router_prompt):
                        yield chunk
                    self.current_state = "continue_counsellor"

                elif "survey" in state:
                    async for chunk in self.handler.agenerate_survey_response(
                        user_input, chat_history, args, self.username, self.router_prompt):
                        yield chunk
                    self.current_state = "continue_survey"

                elif "RAG_NS" in state:
                    async for chunk in self.handler.agenerate_rag_ns_response(
                        user_input, chat_history, args, self.router_prompt):
                        yield chunk
                    self.current_state = "continue_RAG_NS"
                elif "empathetic" in state:
                    async for chunk in self.handler.agenerate_empathetic_response(
                        user_input, chat_history, args, self.router_prompt):
                        yield chunk
                    self.current_state = "continue_empathetic"
                # Fallback router : check state using semantic router if undetermined by llm router
                else:
                    yield {'log': 'Replanning...\n'}
                    logger.info(f"LLM state undecided by planner. Falling back to semantic router")
                    route = self.router(user_input)
                    if route.name == "Counsellor":
                        yield {'log': f'Thinking...\n'}
                        async for chunk in self.handler.agenerate_counsellor_response(
                            user_input, chat_history, args, self.router_prompt):
                            yield chunk
                        self.current_state = "continue_counsellor"
                    elif route.name == "Survey":
                        yield {'log': f'Thinking...\n'}
                        async for chunk in self.handler.agenerate_survey_response(
                            user_input, chat_history, args, self.username, self.router_prompt):
                            yield chunk
                        self.current_state = "continue_survey"
                    elif route.name == "RAG_NS":
                        yield {'log': f'Thinking...\n'}
                        async for chunk in self.handler.agenerate_rag_ns_response(
                            user_input, chat_history, args, self.router_prompt):
                            yield chunk
                        self.current_state = "continue_RAG_NS"
                    else:
                        yield {'log': f'Thinking...\n'}
                        async for chunk in self.handler.agenerate_empathetic_response(
                            user_input, chat_history, args, self.router_prompt):
                            yield chunk
                        self.current_state = "continue_empathetic"

