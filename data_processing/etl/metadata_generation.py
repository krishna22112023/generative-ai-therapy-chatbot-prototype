import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import json
import requests
from core import utils
import ast
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def get_existing_content(output_file):
    try:
        with open(output_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_content(content, output_file):
    with open(output_file, 'w') as f:
        json.dump(content, f, indent=4)

def generate_metadata(system_prompt, user_prompt, model="gemma2:27b", stream=False):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": stream
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        if stream:
            def response_generator():
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)["response"]
            return response_generator()
        else:
            result = response.json()
            return result["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return None

def standardize_content(json_file_paths, output_file):
    generate_keywords_sys_prompt = utils.read_text_file("./data/prompts/metadata_gen/keywords_gen.txt")
    generate_name_sys_prompt = utils.read_text_file("./data/prompts/metadata_gen/name_gen.txt")
    generate_description_sys_prompt = utils.read_text_file("./data/prompts/metadata_gen/descriptions_gen.txt")

    existing_content = get_existing_content(output_file)
    last_processed_index = existing_content[-1]['ID'] if existing_content else -1
    logger.info(f"Resuming from index {last_processed_index + 1}")

    for files in json_file_paths:
        if "counselor" in files:
            cat0 = "counselor_RAG"
        elif "NS" in files:
            cat0 = "NS_RAG"
        else:
            cat0 = "Other"

        with open(files) as f:
            json_content = json.load(f)
        logger.info(f"Found {len(json_content)} records in the json file")

        for index, item in tqdm(enumerate(json_content), total=len(json_content)):
            if index <= last_processed_index:
                continue

            try:
                logger.info(f"Processing index {index}")

                # Generate name
                if item.get("name", "") == "":
                    generate_name_user_prompt = item.get("text", "")
                    name = generate_metadata(generate_name_sys_prompt, generate_name_user_prompt)
                    name = name.strip() if name else ""
                else:
                    name = item.get("name", "")

                # Generate description
                if item.get("description", "") == "":
                    generate_description_user_prompt = item.get("text", "")
                    description = generate_metadata(generate_description_sys_prompt, generate_description_user_prompt)
                    description = description.strip() if description else ""
                else:
                    description = item.get("description", "")

                # Generate keywords
                generate_keywords_user_prompt = f"{name}\n{description}\n{item.get('text', '')}"
                response = generate_metadata(generate_keywords_sys_prompt, generate_keywords_user_prompt)
                response = response.strip() if response else "[]"
                keywords = ast.literal_eval(response)

                content = {
                    "ID": index,
                    "Name": name,
                    "description": description,
                    "response_type": item.get("response_type", ""),
                    "keywords": keywords,
                    "cat0": cat0,
                    "text": item.get("text", ""),
                    "verbatim_flag": False,
                    "url": item.get("url", ""),
                    "duration": item.get("duration", "")
                }

                existing_content.append(content)
                save_content(existing_content, output_file)  # Save after each new item is processed

            except Exception as e:
                logger.error(f"An error occurred while processing item {index}: {e}")
                continue  # Continue with the next item

if __name__ == '__main__':
    json_paths = ["./data/scraped/counselor/crawled_data_processed.json"]
    output_file = "./data/processed/processed_counselor.json"

    standardize_content(json_paths, output_file)