from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from run_with_cache import chat, achat
from general import utils
import os
import logging
import traceback

import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
from conf.config import cfg

logger = logging.getLogger(__name__)

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    chat_id: str
    username: str


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    usage: dict
    choices: List[dict]
    state: str

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration for scraping.")
utils.setup_logging(
    logging_config_path=os.path.join(
        base_path, "conf/logging.yaml"
    )
)

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    try:
        print("request: ", request)
        username = request.username
        session_id = request.chat_id
        # Extract the last user message
        user_message = request.messages[-1].content if request.messages else ""
        print(f"input request {request}")
        print(f"user message {user_message}")

        # Call the run function from run_openwebui.py
        try:
            result = chat(username, session_id, user_message)
        except Exception as e:
            formatted_exception = traceback.format_exc()
            logger.error(f"buddyaid completion error {e}", formatted_exception)
        result_dict = json.loads(result)

        # Prepare the response
        response = ChatResponse(
            id="chatcmpl-" + str(hash(result_dict["response"]))[:8],
            object="chat.completion",
            created=int(__import__("time").time()),
            model="buddyaid-model",
            usage={
                "prompt_tokens": len(user_message),
                "completion_tokens": len(result_dict["response"]),
                "total_tokens": len(user_message) + len(result_dict["response"]),
            },
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": result_dict["response"],
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            state=result_dict["state"],
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/chat/completions")
async def chat_completion(request: ChatRequest):
    username = request.username
    session_id = request.chat_id
    user_message = request.messages[-1].content if request.messages else ""
    print(f"input request {request}")
    print(f"user message {user_message}")

    return StreamingResponse(achat(username, session_id, user_message), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
