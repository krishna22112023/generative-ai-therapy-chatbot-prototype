import json
import re
import pytz
import os
import json
import logging
from pydantic import BaseModel, create_model
from typing import Dict, Any
from conf.config import cfg
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from conf.config import system_config

logger = logging.getLogger(__name__)

from datetime import datetime, timedelta


def read_text_file(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logger.info("The file was not found.")
        return None
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None


def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("File not found. Please check the file path and try again.")
        return None
    except json.JSONDecodeError:
        logger.error("Error decoding JSON. Please check the file content.")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


def load_bio(filepath, username):
    """
    Loads user data from a given JSON file and converts it to a plain text string.

    Parameters:
        filepath (str): The path to the JSON file.
        username (str): The username to look for in the data.

    Returns:
        str: A string representation of the user's bio data, or an error message.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Attempt to fetch user data
        user_data = data.get(username)
        if not user_data:
            return f"User {username} not found."

        # Convert user data to text
        bio_text = ""
        if isinstance(user_data, list):
            for item in user_data:
                bio_text += "".join(f"{key}: {value}\n " for key, value in item.items())
        else:
            bio_text += "".join(
                f"{key}: {value}\n " for key, value in user_data.items()
            )

        return bio_text

    except FileNotFoundError as e:
        logger.error(f"An error occurred when loading bio {e}")
        return "The file was not found."
    except json.JSONDecodeError as e:
        logger.error(f"An error occurred when loading bio {e}")
        return "Error decoding the JSON file."


def load_facts(filepath, username):
    try:
        with open(filepath) as f:
            user_persona = json.load(f)
        user_persona_prompt = user_persona[username][0]["facts"]
        return user_persona_prompt
    except Exception as e:
        logger.error(f"An error occurred when loading user facts {e}")
        return None
    

def load_survey_due(filepath):
    with open(filepath) as f:
        survey = json.load(f)
    return survey

def extract_survey_response(text):
    """
    Extracts content within specified tags using regex and saves each in a separate variable.

    Args:
    text (str): The input text containing tagged content.

    Returns:
    tuple: A tuple containing the extracted content in the order:
           (thinking,state,response)

    """
    patterns = {
        "thinking": r"<thinking>(.*?)</thinking>",
        "state": r"<state>(.*?)</state>",
        "response": r"<response>(.*?)</response>",
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        results[key] = match.group(1).strip() if match else None

    return results

def extract_counsellor_content(text):
    """
    Extracts content within specified tags using regex and saves each in a separate variable.

    Args:
    text (str): The input text containing tagged content.

    Returns:
    tuple: A tuple containing the extracted content in the order:
           (thinking,state,response)

    """
    patterns = {
        "thinking": r"<thinking>(.*?)</thinking>",
        "state": r"<state>(.*?)</state>",
        "save_event": r"<save event>(.*?)</save event>",
        "response": r"<response>(.*?)</response>",
        "turn_count": r"<turn_count>(.*?)</turn_count>",
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        results[key] = match.group(1).strip() if match else None

    return results


def extract_router_states(text):
    """
    Extracts content within specified tags using regex and saves each in a separate variable.

    Args:
    text (str): The input text containing tagged content.

    Returns:
    tuple: A tuple containing the extracted content in the order:
           (thinking,state,response)

    """
    patterns = {
        "thinking": r"<thinking>(.*?)</thinking>",
        "state": r"<state>(.*?)</state>",
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        results[key] = match.group(1).strip() if match else None

    return results


def extract_NS_RAG_content(text):
    """
    Extracts content within specified tags using regex and saves each in a separate variable.

    Args:
    text (str): The input text containing tagged content.

    Returns:
    tuple: A tuple containing the extracted content in the order:
           (thinking,state,response)

    """
    patterns = {
        "thinking": r"<thinking>(.*?)</thinking>",
        "state": r"<state>(.*?)</state>",
        "save_event": r"<save event>(.*?)</save event>",
        "response": r"<response>(.*?)</response>",
        "turn_count": r"<turn_count>(.*?)</turn_count>",
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        results[key] = match.group(1).strip() if match else None

    return results


def extract_empathetic_content(text):
    """
    Extracts content within specified tags using regex and saves each in a separate variable.

    Args:
    text (str): The input text containing tagged content.

    Returns:
    tuple: A tuple containing the extracted content in the order:
           (final_thought, final_response)

    """
    patterns = {
        "thinking": r"<thinking>(.*?)</thinking>",
        "state": r"<state>(.*?)</state>",
        "save_event": r"<save event>(.*?)</save event>",
        "response": r"<response>(.*?)</response>",
        "turn_count": r"<turn_count>(.*?)</turn_count>",
    }
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        results[key] = match.group(1).strip() if match else None

    return results


def dict2xml(result):
    total = ""
    for k, v in result.items():
        info = f"<{k}>{v}</{k}>"
        total += info
    return total


def format_chat_history(chat_history: list) -> list:
    formatted_history = []

    for message in chat_history:
        if message["role"] == "user":
            if message["content"] is not None:
                formatted_history.append(HumanMessage(content=message["content"]))

        elif message["role"] == "assistant":
            if message["content"] is not None:
                formatted_history.append(AIMessage(content=message["content"]))

    return formatted_history


def get_chat_model():
    llm = None
    model_type = system_config["model_type"]
    if model_type == 'openai':
        llm = ChatOpenAI(
            model_name=system_config.get("model_name"),
            temperature=system_config.get("temperature")
        )
    elif model_type == 'groq':
        llm = ChatGroq(
            model_name=system_config.get("model_name"),
            temperature=system_config.get("temperature")
        )
    else:
        llm = ChatOpenAI(
            model_name=system_config.get("model_name"),
            temperature=system_config.get("temperature")
        )
    return llm

def save_json_data(data, file_path):
    """Save the given data as JSON in the specified file."""
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except IOError:
        logger.info("Error writing to file. Please check file permissions and path.")


def save_survey_responses(survey_answer, filepath, username, survey_name):
    survey_answer_string = str(survey_answer)
    answers = re.findall(r"Answer\d+='([^']*)'", survey_answer_string)

    sg_timezone = pytz.timezone("Asia/Singapore")
    date_str = datetime.now(sg_timezone).date().isoformat()
    formatted_data = {date_str: [dict(enumerate(answers, start=1))]}

    existing_data = load_json_data(filepath)

    if username not in existing_data:
        existing_data[username] = {}
    if survey_name not in existing_data[username]:
        existing_data[username][survey_name] = {}
    existing_data[username][survey_name].update(formatted_data)

    save_json_data(existing_data, filepath)
    logging.info(
        f"Saved survey responses for user {username}, survey {survey_name}, date {date_str}"
    )


def load_events(filepath, username):
    """
    Load events for a specific user from a JSON file.

    Args:
    filepath (str): Path to the JSON file containing events.
    username (str): Username to load events for.

    Returns:
    dict: Events for the specified user, or an empty dict if not found.
    """
    if not os.path.exists(filepath):
        logger.info(f"Events file not found at {filepath}")
        return {}

    try:
        with open(filepath, "r") as f:
            user_events = json.load(f)

        return user_events.get(username, {})

    except json.JSONDecodeError:
        logger.info(f"Invalid JSON in file {filepath}")
        return {}
    except Exception as e:
        logger.info(f"Error loading events from {filepath}: {str(e)}")
        return {}


def extract_save_event(response):
    """
    Extracts the content within the `<save event>` and `</save event>` tags from the given response string.

    Parameters:
    response (str): The input string containing the content to be extracted.

    Returns:
    str or None: The extracted content within the `<save event>` and `</save event>` tags.
    Returns None if no match is found or if an error occurs.
    """
    try:
        if not isinstance(response, (str, bytes)):
            logging.warning(f"Invalid response type: {type(response)}. Expected string or bytes.")
            return None
        pattern = r"<save event>(.*?)</save event>"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            event = match.group(1).strip()
            return event
        else:
            logging.info("No <save event> tags found in the response.")
        return None

    except Exception as e:
        logging.error(f"Error in extract_save_event: {str(e)}")
    return None


def check_gad7_due_status(username, days_threshold, filepath):
    """
    Checks if the GAD7 submission for a user is due based on the latest submission date and calculates the number of days
    until or since the GAD7 is due.

    Parameters:
        username (str): The username to check the GAD7 status for.
        days_threshold (int): The number of days to check against for due status.
        filepath (str): The path to the JSON file containing GAD7 answers.

    Returns:
        int: The number of days until or since the GAD7 survey is due. Positive if due in the future, negative if overdue.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        user_data = data.get(username)
        if not user_data:
            return None  # Return None if user data is not found

        latest_date_str = max(user_data.keys())
        latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
        current_date = datetime.now()

        due_date = latest_date + timedelta(days=days_threshold)
        days_until_due = (due_date - current_date).days

        return days_until_due
        # Return the number of days until due (positive or negative)

    except FileNotFoundError as f:
        logger.info(f"Error reading file gad7_answers {f}")
        return None  # Return None if the file is not found
    except json.JSONDecodeError as f:
        logger.info(f"Corrupted file gad7_answers {f}")
        return None  # Return None if there is a decoding error

def check_surveys_due(username, days_thresholds, filepath):
    """
    Checks survey submission to see if a user is due based on the latest submission date and calculates the number of days.

    Parameters:
        username (str): The username to check the surveys status for.
        days_thresholds (dict): Survey names and days thresholds to check the due status.
        filepath (str): The path to the JSON file containing survey answers.

    Returns:
        dict: A dictionary containing the due status and days since due for each survey type.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        user_data = data.get(username)
        if not user_data:
            logger.error(f"User '{username}' not found in data.")
            return None

        result = {}
        current_date = datetime.now().date()

        for survey_type, threshold in days_thresholds.items():
            if survey_type in user_data:
                survey_dates = user_data[survey_type].keys()
                if survey_dates:
                    latest_date = max(
                        datetime.strptime(date, "%Y-%m-%d").date()
                        for date in survey_dates
                    )
                    due_date = latest_date + timedelta(days=threshold)
                    days_difference = (due_date - current_date).days

                else:
                    due_date = current_date
                    days_difference = -99  # First time taking the survey
            else:
                due_date = current_date
                days_difference = -99  # First time taking the survey

            is_due = days_difference <= 0

            if days_difference == -99:
                message = f"Your survey is due today. This is the first time you will take this survey."
            elif days_difference == 0:
                message = f"Your next survey is due today."
            elif days_difference == 1:
                message = f"Your next survey is due tomorrow."
            elif days_difference > 1:
                message = f"Your next survey is due in {days_difference} days."
            else:
                message = f"Survey is {abs(days_difference)} day(s) overdue."

            result[survey_type] = {
                "is_due": is_due,
                "due_date": due_date.strftime("%Y-%m-%d"),
                "days_difference": days_difference,
                "message": message,
            }

        logger.info(f"Successfully checked surveys due for user '{username}'")
        return result

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {type(e).__name__}: {str(e)}")
        return None


def get_max_days_surveys_due(survey_due_results):
    """
    Takes the output of check_surveys_due and returns the maximum days_difference.

    Parameters:
        survey_due_results (dict): The dictionary returned by check_surveys_due.

    Returns:
        int: The maximum days_difference value, or None if the input is invalid.
    """
    if not survey_due_results or not isinstance(survey_due_results, dict):
        logger.error("Invalid input to get_max_days_difference")
        return None

    try:
        max_survey_due_days = min(
            survey["days_difference"]
            for survey in survey_due_results.values()
            if isinstance(survey, dict) and "days_difference" in survey
        )
        return max_survey_due_days
    except (
        ValueError
    ):  # This would occur if the max() function receives an empty sequence
        logger.error("No valid days_difference values found in the input")
        return None


def load_survey_name_due(survey_due_results):
    if not survey_due_results or not isinstance(survey_due_results, dict):
        logging.warning("Invalid input to load_survey_name_due")
        return None

    # Find the survey with the lowest days_difference
    most_imminent_survey = min(
        survey_due_results.items(),
        key=lambda x: (
            x[1]["days_difference"]
            if isinstance(x[1], dict) and "days_difference" in x[1]
            else float("inf")
        ),
    )

    logging.info(f"Most imminent survey: {most_imminent_survey[0]}")
    return most_imminent_survey[0]


def create_dynamic_survey_answer(survey_structure: Dict[str, Any]) -> type[BaseModel]:
    fields = {}
    for question_num in range(
        1, int(survey_structure["info"][0]["Total_Questions"]) + 1
    ):
        fields[f"Question{question_num}"] = (str, ...)
        answer_type = survey_structure["answer_structure"][0][f"Answer{question_num}"]
        if answer_type == "int":
            fields[f"Answer{question_num}"] = (int, ...)
        else:
            fields[f"Answer{question_num}"] = (str, ...)

    return create_model("DynamicSurveyAnswer", **fields)


def score_and_interpret_latest_survey(
    username, survey_name, filepath=cfg.survey.survey_answers
):
    """
    Scores the latest survey answers from a JSON file, interprets the results, and provides feedback.
    Includes a flag indicating whether scoring is available for the survey.
    If scoring is not available or there's an error, score-related fields are set to None.

    Parameters:
    - username (str): The username of the participant to score.
    - survey_name (str): The name of the survey (e.g., 'gad7', 'phq9').
    - filepath (str): Path to the JSON file containing survey answers.

    Returns:
    - dict: A dictionary containing the survey date, raw answers, score availability, and score/interpretation if available.
             In case of an error, it includes an error message and sets other fields to None or appropriate default values.
    """
    try:
        # Load survey answers
        with open(filepath, "r") as file:
            data = json.load(file)

        # Check if user and survey data exist
        user_data = data.get(username, {})
        survey_data = user_data.get(survey_name, {})

        if not survey_data:
            logger.error(
                f"No survey data found for user {username} for survey {survey_name}"
            )
            return {
                "error": "No survey data found",
                "survey_date": None,
                "raw_answers": {},
                "score_available": False,
                "score": None,
                "interpretation": None,
                "survey_info": None,
            }

        # Find the latest survey date
        latest_date = max(
            survey_data.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d")
        )
        date_data = survey_data[latest_date]

        result = {
            "error": None,
            "survey_date": latest_date,
            "raw_answers": date_data[
                0
            ],  # Assuming there's only one set of answers per date
            "score_available": False,
            "score": None,
            "interpretation": None,
            "survey_info": None,
        }

        # Check if scoring file exists
        #scoring_file_path = f"data/raw/survey/survey_{survey_name}_scoring.json"
        scoring_file_path = getattr(cfg.survey, f"survey_{survey_name}_scoring")
        logger.info(f"scoring file path {scoring_file_path}")
        if not os.path.exists(scoring_file_path):
            logger.error(
                f"No scoring file found for {survey_name}. Returning raw answers only."
            )
            return result

        # Load scoring data
        with open(scoring_file_path, "r") as file:
            scoring_data = json.load(file)

        # Extract scoring mechanism and results
        scoring_mechanism = scoring_data.get("scoring", [])
        results = scoring_data.get("results", [])

        # Score the survey
        total_score = 0
        for question, answer in result["raw_answers"].items():
            question_scoring = scoring_mechanism[0].get(question, "")
            score_mapping = dict(
                item.split(":") for item in question_scoring.split(";")
            )
            total_score += int(score_mapping.get(answer, 0))

        # Interpret the score
        interpretation = "No interpretation available."
        for result_range in results[0]:
            min_score, max_score = map(int, result_range.split("-"))
            if min_score <= total_score <= max_score:
                interpretation = results[0][result_range]
                break

        result.update(
            {
                "score_available": True,
                "score": total_score,
                "interpretation": interpretation,
                "survey_info": scoring_data.get("info", [{}])[0],
            }
        )

        return result

    except Exception as e:
        logger.error(f"Error processing survey: {str(e)}")
        return {
            "error": str(e),
            "survey_date": None,
            "raw_answers": {},
            "score_available": False,
            "score": None,
            "interpretation": None,
            "survey_info": None,
        }


def save_scored_survey(
    username, survey_name, scored_result, filepath=cfg.survey.survey_scored
):
    """
    Saves only the score of the survey results to a JSON file, if a score is available.

    Parameters:
    - username (str): The username of the participant.
    - survey_name (str): The name of the survey (e.g., 'gad7', 'phq9').
    - scored_result (dict): The dictionary containing the scored and interpreted survey results.
    - filepath (str): Path to the JSON file where scored results will be saved.

    Returns:
    - bool: True if the score was saved, False otherwise.
    """
    if not scored_result.get("score_available", False):
        logger.info(
            f"Score not available for user {username}, survey {survey_name}. Not saving."
        )
        return False

    survey_date = scored_result.get("survey_date")
    score = scored_result.get("score")

    if not survey_date or score is None:
        logger.error(
            f"Survey date or score not found in scored result for user {username}, survey {survey_name}"
        )
        return False

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Load existing data or create new data structure
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    logger.error(
                        f"Existing file {filepath} is not valid JSON. Creating new data structure."
                    )
                    data = {}
        else:
            data = {}

        # Ensure nested structure exists
        if username not in data:
            data[username] = {}
        if survey_name not in data[username]:
            data[username][survey_name] = {}

        # Save only the score
        data[username][survey_name][survey_date] = score

        # Write the updated data back to the file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

        logger.info(
            f"Survey score saved for user {username}, survey {survey_name}, date {survey_date}"
        )
        return True
    except Exception as e:
        logger.error(f"Error saving survey score: {str(e)}", exc_info=True)
        return False

import requests
from pathlib import Path
import uuid

def download_gdrive_image(gdrive_url, save_dir):
    # Extract the file ID from the Google Drive URL
    file_id = gdrive_url.split('/d/')[1].split('/')[0]
    
    # Construct the direct download link
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Make a request to the download URL
    response = requests.get(download_url)
    
    if response.status_code == 200:
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.png"
        
        # Construct the full save path
        save_path = Path(save_dir) / filename
        
        # Save the image content
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return str(filename)
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
