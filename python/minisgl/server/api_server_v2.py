from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.response import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    TokenizeMsg,
    UserReply
)

from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs
logger = init_logger(__name__, "FrontendAPI")
_GLOBAL_STATE = None

def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE

def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]: 
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
    assert isinstance(msg, UserReply)
    return [msg]
    


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: ['system', 'user', 'assistant']
    content: str

class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = -1
    top_p: flaot = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(defualt_factory=lambda: int(time.time()))
    owned_by: str = 'mini-sgl'
    root: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            aysncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    # 异步生成器，用于等待确认消息
    async def await_for_ack(self, uid: int):
        # 获取事件对象
        event = self.event_map[uid]
        while True:
            # 等待事件被设置
            await event.wait()
            # 清除事件状态，准备接受下一批
            event.clear()

            # 获取所有待处理的消息
            pending = self.ack_map[uid]
            # 清空待处理
            self.ack_map[uid] = []
            ack = None
            # 遍历所有待处理消息
            for ack in pending:
                yield ack # 逐个流式输出

            # 完成就退出
            if ack and ack.finished:
                break
        # 删除用户消息和事件
        del self.ack_map[uid]
        del self.event_map[uid]


    # 异步生成器，用于生成符合Server-Sent Events（SSE）格式的响应
    """
    uid: user id
    """
    async def stream_generate(self, uid: int):
        # 等待并处理确认消息
        async for ack in self.wait_for_ack(uid):
            # 生成SSE数据格式，并完成条件检查
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        # 发送完成标记
        yield "data: [Done]\n".encode()
        # 日记记录
        logger.debug("finished streaming response for user %s", uid)

    # 一个异步生成器函数，专门用于生成符合OpenAI Chat Completions API的响应，实现了完整的OpenAI流式聊天协议

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            data = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                'choices': [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break
        
        # send finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [Done]\n\n"
        logger.debug("finished streaming response for user %s", uid)
            


    async def stream_with_cancelllation(self, generator, request: Request, uid: int):
        pass

    async def abort_user(self, uid: int):
        pass

    def shutdown(self):
        pass



