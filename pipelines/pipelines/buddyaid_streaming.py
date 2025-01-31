import json
import requests
from typing import List, Union, Generator, Iterator, Optional


class Pipeline:
    def __init__(self):
        self.name = 'BuddyStream'
        self.api_base = "http://127.0.0.1:8000"

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        body['meta'] = {'session_id': body['session_id'], 'chat_id': body['chat_id']}
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[
        str, Generator, Iterator]:
        headers = {"Content-Type": "application/json"}

        data = {
            "username": body["user"]["name"],
            "user_id": 'U0003',
            "session_id": body['meta']["chat_id"],
            "chat_id": body['meta']["chat_id"],
            "messages": messages,
            "stream": True
        }

        payload = json.dumps(data)

        r = requests.post(f"{self.api_base}/v2/chat/completions", data=payload, headers=headers, stream=True)
        r.raise_for_status()
        for chunk in r:
            print(chunk)
            yield chunk