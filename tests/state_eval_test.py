import sys
from pathlib import Path
import json
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datetime import datetime
from core.response_llm import response_validation
from core.logic import NationalServiceLogic, LLMTransitionLogic
from core.state import NationalServiceApp
from core.utils import load_json_data

def evaluate_responses(data, username, args):
    results = {"easy": [], "medium": [], "hard": []}
    generated_states_info = {"easy": [], "medium": [], "hard": []}
    for difficulty, prompts in data["eval_dataset"].items():
        for prompt in tqdm(prompts,desc=f'evaluating {difficulty} task prompt'):
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
            actual_states = prompt["actual_states"]
            chat_history = []
            generated_states = []
            for p in prompt["task_prompt"]:
                logger.info(f"input prompt: {p}")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                day_of_week = datetime.now().strftime("%A")
                preamble = f"[Additional Information: Current Date Time is {current_time}, Day of the Week: {day_of_week}] \n "
                full_user_query = preamble + p
                chat_history.append({"role": "user", "content": full_user_query})
                try:
                    response, state = llm_router.gen_response(full_user_query, chat_history, args) 
                except Exception as e:
                    logger.info("an error occured while generating a response. State = None.")
                    state = ""
                    response = ""
                generated_states.append(state)
                if isinstance(response, tuple):
                    response = response[0]
                chat_history.append({"role": "assistant", "content": response})
            logger.info(f"chat history: {chat_history}")
            matches = sum(a in g for a, g in zip(actual_states, generated_states))
            logger.info(f"matches found : {matches}")
            results[difficulty].append(matches / len(actual_states))
            generated_states_info[difficulty].append({
                "idx": prompt["idx"],
                "actual_states": actual_states,
                "generated_states": generated_states
            })
    accuracy_results = {level: sum(scores) / len(scores) if scores else 0 
            for level, scores in results.items()}

    return accuracy_results, generated_states_info


def main():
    # arguments
    username = "benjamin.davis"
    args = {
        "model_name": "gpt-4o",
        "temperature": 1,
        "streaming": True,
        "days_thresholds": {
            "gad7": 2,
            "phq9": 1,
            "demographics": 7,
            "demographics2": 7,
        },
        "survey_answers_file_path": "./data/raw/survey/survey_answers.json",
        "embedding": {"type": "OpenAIEmbeddings", "model": "text-embedding-ada-002"},
        "search_type": "hybrid",
        "top_k": 5,
        "search_weight": 0.5,
        "metadata_filter_NS": {"cat0": "NS_RAG"},
        "metadata_filter_counselor": {"cat0": "counselor_RAG"},
    }
    task_types = ["single_task", "double_task", "triple_task"]
    final_results = []
    final_generated_states = []

    for task_type in task_types:
        logger.info(f"evaluating task type : {task_type}")
        file_path = f"./data/eval/state_transition/{task_type}.json"
        data = load_json_data(file_path)
        results, generated_states = evaluate_responses(data,username,args)
        final_results.append({task_type: results})
        final_generated_states.append({task_type: generated_states})

    with open("./data/eval/state_transition/evaluation_results_revised.json", "w") as outfile:
        json.dump(final_results, outfile, indent=2)
    
    with open("./data/eval/state_transition/generated_states_revised.json", "w") as outfile:
        json.dump(final_generated_states, outfile, indent=2)

    logger.info("Evaluation complete. Results saved in ./data/eval/state_transition/evaluation_results.json'")

    
if __name__ == "__main__":
    main()
