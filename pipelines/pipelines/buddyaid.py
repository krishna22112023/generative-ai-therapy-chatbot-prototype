"""
title: BuddyAid Pipeline
author: open-webui
date: 2024-10-03
version: 1.0
license: MIT
description: A pipeline for interacting with the BuddyAid API and managing conversation state.
requirements: requests
"""

import requests
import re
from typing import List, Union, Generator, Iterator, Optional
import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def extract_responses(text):
    pattern1 = r'<response>(.*?)</response>'
    responses = re.findall(pattern1, text, re.DOTALL)
    pattern2 = r'<sources>(.*?)</sources>'
    sources = re.findall(pattern2, text, re.DOTALL)
    if len(sources)>0:
        return responses[0]+'\n'+sources[0]
    else:
        return responses[0]

class Pipeline:
    def __init__(self):
        self.api_base = "http://localhost:8000"  

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
    
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        body["pipeline_metadata"] = {"chat_id": body["chat_id"]}
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where we interact with the BuddyAid API
        print(f"user message: {user_message}")
        print(f"model id: {model_id}")
        print(f"body : {body}")
        headers = {"Content-Type": "application/json"}

        # Prepare the request data
        data = {
            "messages": messages,
            "chat_id": body["pipeline_metadata"]["chat_id"],
            "username": body["user"]["name"]
        }
        #data = json.dumps(data)
        print(f"input data sent to pipeline API {data}")
        # Make a request to the BuddyAid API
        response = requests.post(f"{self.api_base}/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status() 
        result = response.json()
        print(f"pipeline response after converting to json {result}")

        # Extract the assistant's message and new state
        assistant_message = result["choices"][0]["message"]["content"]

        assistant_message = extract_responses(assistant_message)
        # Yield the assistant's message
        return assistant_message

