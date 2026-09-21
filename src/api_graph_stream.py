import re
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graph.build import build_graph
from src.tts.kokoro_provider import KokoroTTSProvider

app = FastAPI(title="LangGraph Agent Streaming to Kokoro TTS")

graph = build_graph()
tts_provider = KokoroTTSProvider()

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\n])\s+")


class AgentQueryRequest(BaseModel):
    query: str
    voice: str = "af_heart"
    thread_id: str = "default_session"


async def stream_graph_sentences(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    initial_input = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""

    async for chunk, metadata in graph.astream(
        initial_input,
        config=config,
        stream_mode="messages",
    ):
        content = chunk.content if hasattr(chunk, "content") else ""
        if not content or not isinstance(content, str):
            continue

        buffer += content
        parts = SENTENCE_SPLIT_REGEX.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                clean = sentence.strip()
                if clean:
                    yield clean
            buffer = parts[-1]

    remaining = buffer.strip()
    if remaining:
        yield remaining


async def stream_audio_chunks(query: str, voice: str, thread_id: str) -> AsyncGenerator[bytes, None]:
    synth_fn = getattr(tts_provider, "synthesize", getattr(tts_provider, "create_audio", None))
    async for sentence in stream_graph_sentences(query, thread_id):
        if synth_fn:
            audio_bytes = await asyncio.to_thread(synth_fn, sentence, voice)
            if audio_bytes:
                yield audio_bytes


@app.post("/agent/stream-text")
async def agent_text_endpoint(req: AgentQueryRequest):
    return StreamingResponse(
        stream_graph_sentences(req.query, req.thread_id),
        media_type="text/plain",
    )


@app.post("/agent/stream-audio")
async def agent_audio_endpoint(req: AgentQueryRequest):
    return StreamingResponse(
        stream_audio_chunks(req.query, req.voice, req.thread_id),
        media_type="audio/pcm",
    )
