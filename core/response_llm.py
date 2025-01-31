from core.event_manager import EventManager
from semantic_layer import LLM_router
from core.utils import read_text_file
import logging
from conf.config import cfg


class ResponseHandler:
    """
    Handles the logic to generate responses based on the National Service stage of the user.
    """

    def __init__(self, ns_state, username, llm_states):
        """
        Initializes the response handler with the username and checks the NSAppState

        Args:
        username (str): The username of the national service member.
        """
        self.username = username
        self.ns_state = ns_state

        self.state = {
            "last_state": None,
            "turn_count": 1,
            "save_event": None
        }

        if ns_state == "duringNS":
            (
                self.empathetic_state,
                self.counsellor_state,
                self.survey_state,
                self.ns_rag_state,
            ) = llm_states
        else:
            self.empathetic_state, self.counsellor_state, self.ns_rag_state = llm_states

    def update_survey_state(self, survey_name):
        if hasattr(self, "survey_state"):
            self.survey_state.update_survey_info(survey_name)
        else:
            logging.warning("Survey state not available in current NS state")

    def generate_empathetic_response(
        self, user_input, chat_history, args, router_prompt=None
    ):
        """
        Generates an empathetic response based on the user's input and the current NS stage.

        Args:
        user_input (str): The input provided by the user.

        Returns:
        str: The generated response from the chatbot.
        """
        # Enter the state to load necessary data and format the initial prompt
        formatted_prompt = self.empathetic_state.enter()

        # Execute the state to generate a response based on the formatted prompt and user input
        structured_response, raw_response = self.empathetic_state.execute(
            formatted_prompt, user_input, chat_history, args
        )
        if "continue_empathetic" in structured_response.get(
            "state", "continue_empathetic"
        ):
            self.state["last_state"] = "continue_empathetic"
        elif "end_empathetic" in structured_response.get("state", "end_empathetic"):
            self.state["last_state"] = "end_empathetic"
            self.empathetic_state.end()

            # Transition to correctly to next llm if required
            template = read_text_file(router_prompt)
            LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(
                template
            )
            Router_llm = LLM_router.chat_completion.RouterLLM(args)
            llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
            if "survey" in llm_route.get("state"):
                raw_response = self.generate_survey_response(
                    user_input, chat_history, args, self.username
                )
            elif "RAG_NS" in llm_route.get("state"):
                raw_response = self.generate_rag_ns_response(
                    user_input, chat_history, args
                )
            elif "counselor" in llm_route.get("state"):
                raw_response = self.generate_counsellor_response(
                    user_input, chat_history, args
                )
        else:
            self.state["last_state"] = "continue_empathetic"
        self.state["turn_count"] = int(structured_response.get("turn_count", 1) if structured_response else 1)
        self.state["save_event"] = structured_response.get("save_event", "NA" if structured_response else 'NA')
        return raw_response

    def generate_counsellor_response(
        self, user_input, chat_history, args, router_prompt=None
    ):
        """
        Generates an coping strategies based on the user's input and the current NS stage.

        Args:
        user_input (str): The input provided by the user.

        Returns:
        str: The generated response from the chatbot.
        """

        # Enter the state to load necessary data and format the initial prompt
        formatted_prompt = self.counsellor_state.enter()

        # Execute the state to generate a response based on the formatted prompt and user input
        structured_response, raw_response, urls = self.counsellor_state.execute(
            formatted_prompt, user_input, chat_history, args
        )

        if "continue_counsellor" in structured_response.get(
            "state", "continue_counsellor"
        ):
            self.state["last_state"] = "continue_counsellor"
        elif "end_counsellor" in structured_response.get("state", "end_counsellor"):
            self.state["last_state"] = "end_counsellor"
            self.counsellor_state.end()

            # Transition to correctly to next llm if required
            template = read_text_file(router_prompt)
            LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(
                template
            )
            Router_llm = LLM_router.chat_completion.RouterLLM(args)
            llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
            if "survey" in llm_route.get("state"):
                raw_response = self.generate_survey_response(
                    user_input, chat_history, args, self.username
                )
            elif "RAG_NS" in llm_route.get("state"):
                raw_response = self.generate_rag_ns_response(
                    user_input, chat_history, args
                )
            elif "empathetic" in llm_route.get("state"):
                raw_response = self.generate_empathetic_response(
                    user_input, chat_history, args
                )
        else:
            self.state["last_state"] = "continue_empathetic"

        self.state["turn_count"] = int(structured_response.get("turn_count"))
        self.state["save_event"] = structured_response.get("save_event")
        return raw_response, urls

    def generate_rag_ns_response(
        self, user_input, chat_history, args, router_prompt=None
    ):
        """
        Generates an coping strategies based on the user's input and the current NS stage.

        Args:
        user_input (str): The input provided by the user.

        Returns:
        str: The generated response from the chatbot.
        """

        # Enter the state to load necessary data and format the initial prompt
        formatted_prompt = self.ns_rag_state.enter()

        # Execute the state to generate a response based on the formatted prompt and user input
        structured_response, raw_response, urls = self.ns_rag_state.execute(
            formatted_prompt, user_input, chat_history, args
        )

        if "continue_RAG_NS" in structured_response.get("state", "continue_RAG_NS"):
            self.state["last_state"] = "continue_RAG_NS"
        elif "end_RAG_NS" in structured_response.get("state", "end_RAG_NS"):
            self.state["last_state"] = "end_RAG_NS"
            self.ns_rag_state.end()

            # Transition to correctly to next llm if required
            template = read_text_file(router_prompt)
            LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(
                template
            )
            Router_llm = LLM_router.chat_completion.RouterLLM(args)
            llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
            if "survey" in llm_route.get("state"):
                raw_response = self.generate_survey_response(
                    user_input, chat_history, args, self.username
                )
            elif "empathetic" in llm_route.get("state"):
                raw_response = self.generate_empathetic_response(
                    user_input, chat_history, args
                )
            elif "counselor" in llm_route.get("state"):
                raw_response = self.generate_counsellor_response(
                    user_input, chat_history, args
                )
        else:
            self.state["last_state"] = "continue_empathetic"

        self.state["turn_count"] = int(structured_response.get("turn_count","1"))
        return raw_response, urls

    def generate_survey_response(
        self, user_input, chat_history, args, username, router_prompt=None
    ):
        """
        adminsters a pyschological survey for the users

        Args:
        user_input (str): The input provided by the user.

        Returns:
        str: The generated response from the chatbot.
        """
        # Enter the state to load necessary data and format the initial prompt
        formatted_prompt = self.survey_state.enter()

        # Execute the state to generate a response based on the formatted prompt and user input
        structured_response, raw_response = self.survey_state.execute(
            formatted_prompt, user_input, chat_history, args
        )

        # extracting state of the response. If llm fails to return the state variable, then continue survey by default.
        if "end_survey" in structured_response.get("state", "continue_survey"):
            self.state["last_state"] = "end_survey"
            feedback_message = self.survey_state.end(args, self.username, chat_history)
            if feedback_message:
                return feedback_message, True
            else:
                return "Thank you for completing the survey.", True
        else:
            self.state["last_state"] = "continue_survey"
            return raw_response, False

    def check_last_state(self):
        return self.state["last_state"]

    def check_turn_count(self):
        return self.state["turn_count"]

    def check_save_event(self):
        return self.state["save_event"]
    
    def update_state(self, new_state):
        self.state.update(new_state)

    def get_state(self):
        return self.state

    async def agenerate_empathetic_response(self, user_input, chat_history, args, router_prompt=None):
        formatted_prompt = self.empathetic_state.enter()

        final_result = {}
        async for result, text in self.empathetic_state.aexecute(formatted_prompt, user_input, chat_history, args):
            final_result = result
            if 'response' in result and result['response'] != 'NA':
                yield result

        if "continue_empathetic" in final_result.get("state", "continue_empathetic"):
            self.state["last_state"] = "continue_empathetic"
        elif "end_empathetic" in final_result.get("state", "end_empathetic"):
            self.state["last_state"] = "end_empathetic"
            self.empathetic_state.end()

            yield {'log': 'Replanning...\n'}
            template = read_text_file(router_prompt)
            LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(
                template
            )
            Router_llm = LLM_router.chat_completion.RouterLLM(args)
            llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
            state = llm_route.get("state")
            yield {'log': f'Thinking...\n'}

            if "survey" in state:
                async for chunk in self.agenerate_survey_response(
                        user_input, chat_history, args, self.username, cfg.prompts.router):
                    yield chunk

            elif "RAG_NS" in state:
                async for chunk in self.agenerate_rag_ns_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk

            elif "counselor" in state:
                async for chunk in self.agenerate_counsellor_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk

            else:
                yield {'log': 'Whoa there, my AI brain coud not comprehend that! Can you please rephrase that?\n'}

        else:
            self.state["last_state"] = "continue_empathetic"

        self.state["turn_count"] = int(final_result.get("turn_count", 1))
        self.state["save_event"] = final_result.get("save_event", "NA")

    async def agenerate_counsellor_response(
        self, user_input, chat_history, args, router_prompt=None
    ):
        formatted_prompt = self.counsellor_state.enter()

        final_result = {}
        async for result, text, urls in self.counsellor_state.aexecute(formatted_prompt, user_input, chat_history, args):
            final_result = result
            if 'response' in result and result['response'] != 'NA':
                yield result,urls

        if "continue_counsellor" in final_result.get("state", "continue_counsellor"):
            self.state["last_state"] = "continue_counsellor"
        elif "end_counsellor" in final_result.get("state", "end_counsellor"):
            self.state["last_state"] = "end_counsellor"
            self.counsellor_state.end()

            yield {'log': 'Replanning...\n'}
            template = read_text_file(router_prompt)
            LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(template)
            Router_llm = LLM_router.chat_completion.RouterLLM(args)
            llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
            state = llm_route.get("state")
            yield {'log': f'Thinking...\n'}

            if "survey" in state:
                async for chunk in self.agenerate_survey_response(
                        user_input, chat_history, args, self.username, cfg.prompts.router):
                    yield chunk

            elif "RAG_NS" in state:
                yield {'log': f'Thinking..\n'}
                async for chunk in self.agenerate_rag_ns_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk

            elif "empathetic" in state:
                async for chunk in self.agenerate_empathetic_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk

            else:
                async for chunk in self.agenerate_empathetic_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk
        else:
            self.state["last_state"] = "continue_empathetic"

        self.state["turn_count"] = int(final_result.get("turn_count"))
        self.state["save_event"] = final_result.get("save_event")

    async def agenerate_rag_ns_response(self, user_input, chat_history, args, router_prompt=None):
        formatted_prompt = self.ns_rag_state.enter()

        final_result = {}
        async for result, text, urls in self.ns_rag_state.aexecute(formatted_prompt, user_input, chat_history, args):
            final_result = result
            if 'response' in result and result['response']!='NA':
                yield result,urls

        if "continue_RAG_NS" in final_result.get("state", "continue_RAG_NS"):
            self.state["last_state"] = "continue_RAG_NS"
        elif "end_RAG_NS" in final_result.get("state", "end_RAG_NS"):
            self.state["last_state"] = "end_RAG_NS"
            self.ns_rag_state.end()

            yield {'log': '\nReplanning...\n'}
            template = read_text_file(router_prompt)
            LLM_router_prompt = LLM_router.prompt_format.create_llm_router_prompt(template)
            Router_llm = LLM_router.chat_completion.RouterLLM(args)
            llm_route, _ = Router_llm.chat(LLM_router_prompt, user_input)
            state = llm_route.get("state")
            yield {'log': f'Thinking...\n'}

            if "survey" in state:
                async for chunk in self.agenerate_survey_response(user_input, chat_history, args, self.username, cfg.prompts.router):
                    yield chunk

            elif "empathetic" in state:
                async for chunk in self.agenerate_empathetic_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk

            elif "counselor" in state:
                async for chunk in self.agenerate_counsellor_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk

            else:
                async for chunk in self.agenerate_empathetic_response(user_input, chat_history, args, cfg.prompts.router):
                    yield chunk
        else:
            self.state["last_state"] = "continue_empathetic"

        self.state["turn_count"] = int(final_result.get("turn_count","1"))

    async def agenerate_survey_response(
        self, user_input, chat_history, args, username, router_prompt=None
    ):
        formatted_prompt = self.survey_state.enter()

        final_result = {}
        async for result, text in self.survey_state.aexecute(formatted_prompt, user_input, chat_history, args):
            final_result = result
            yield result

        if "end_survey" in final_result.get("state", "continue_survey"):
            self.state["last_state"] = "end_survey"
            feedback_message = self.survey_state.end(args, self.username, chat_history)
            if feedback_message:
                yield feedback_message
            else:
                yield "Thank you for completing the survey.",
        else:
            self.state["last_state"] = "continue_survey"


def response_validation(username, response, full_user_query, chat_history):
    """
    response_validation is a middle layer between response generation and response shown to the user.
    The purpose is to ensure that the response conforms to elements we agreed to (safety, security, etc)
    Also, here we can add any additional processes done on the response, one example is to save the events if events are generated by the model

    Args:
    response (str): The response generated by the chatbot.
    full_user_query (str): The full user query.
    chat_history (list): The chat history.

    Returns:
    str: The response to be shown to the user.
    """
    # Initialize EventManager for the current user
    event_manager = EventManager(username)
    validated_response = response

    # Process and save any events in the response
    saved_event = event_manager.process_event(response)

    if saved_event is not None and saved_event != "NA":
        # If a valid event was saved, add a confirmation to the response
        validated_response += f"\n\nThis event was saved: {saved_event}"

    # Here you can add any other validation or processing steps
    # For example, checking for inappropriate content, ensuring security, etc.

    # Example of a simple safety check (you would replace this with more comprehensive checks)
    # if any(word in response.lower() for word in ["unsafe", "dangerous", "illegal"]):
    #     validated_response = "I apologize, but I cannot provide information about unsafe or illegal activities."

    # You might want to log the interaction
    # logging.info(f"User query: {full_user_query}")
    # logging.info(f"Bot response: \n\n {response}")

    return validated_response


async def demo_async():
    from core.state import NationalServiceApp
    from conf.config import system_config

    username = 'benjamin.davis'

    ns_app = NationalServiceApp(username, system_config)
    llm_states = getattr(ns_app, 'duringNS')()
    model = ResponseHandler("duringNS", username, llm_states)
    user_input = "What is national service in Singapore?"
    async for chunk in model.agenerate_survey_response(user_input, [], system_config, username):
        print(chunk)


def main():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(demo_async())
    loop.close()

if __name__ == '__main__':
    main()