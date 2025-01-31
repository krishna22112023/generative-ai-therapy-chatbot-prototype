import json
import logging
import re
import time
import asyncio
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse

from conf.config import cfg, system_config
from core.response_llm import response_validation
from core.logic import NationalServiceLogic, LLMTransitionLogic
from core.state import NationalServiceApp
from general.utils import Redis
from langsmith import traceable

from run_with_cache import chat

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

    model_config = ConfigDict(
        extra='allow',
    )


traceable()

app = FastAPI(title="OpenAI-compatible API")


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    print('[User Message]: \n', request.messages[-1].content)
    username = request.username
    session_id = request.session_id
    result = chat(username, session_id, f"{request.messages[-1].content}")
    result = json.loads(result)

    content = re.search(r'<response>(.*?)</response>', result["response"], re.DOTALL).group(1).strip()
    state = re.search(r'<state>(.*?)</state>', result["response"], re.DOTALL).group(1).strip()
    turn_count = re.search(r'<turn_count>(.*?)</turn_count>', result["response"], re.DOTALL).group(1).strip()
    print(f'[AI Message (state {state} turn_count {turn_count})]: \n', content)
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": ChatMessage(role="assistant", content=f"{content}")
        }]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.server.host, port=8081)
