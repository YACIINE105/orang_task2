import io
import re
import asyncio
import tempfile
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graph.build import build_graph
from src.tts.kokoro_provider import KokoroTTSProvider
from src.stt.qwen_asr_provider import QwenASRProvider

app = FastAPI(title="LangGraph Agent Streaming to Kokoro TTS & Qwen STT")

graph = build_graph()
tts_provider = KokoroTTSProvider()
stt_provider = QwenASRProvider()

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\n])\s+")


class AgentQueryRequest(BaseModel):
    query: str
    asset_id: str = "22"
    voice: str = "af_heart"
    thread_id: str = "default_session"


async def stream_graph_sentences(query: str, asset_id: str, thread_id: str) -> AsyncGenerator[str, None]:
    initial_input = {
        "query": query,
        "asset_id": asset_id,
        "messages": [{"role": "user", "content": query}],
    }
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""

    async for event in graph.astream(initial_input, config=config):
        text_chunk = ""
        if isinstance(event, dict):
            for _, node_output in event.items():
                if isinstance(node_output, dict):
                    for key in ["response", "final_response", "generation", "output", "summary", "content"]:
                        val = node_output.get(key)
                        if isinstance(val, str) and val.strip():
                            text_chunk += val + " "
                    messages = node_output.get("messages")
                    if isinstance(messages, list) and messages:
                        last_msg = messages[-1]
                        content = getattr(last_msg, "content", None)
                        if isinstance(content, str) and content.strip():
                            text_chunk += content + " "

        if not text_chunk:
            continue

        buffer += text_chunk
        parts = SENTENCE_SPLIT_REGEX.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                clean = sentence.strip()
                if clean:
                    yield clean + "\n\n"
            buffer = parts[-1]

    remaining = buffer.strip()
    if remaining:
        yield remaining + "\n"


async def stream_audio_chunks(query: str, asset_id: str, voice: str, thread_id: str) -> AsyncGenerator[bytes, None]:
    synth_fn = getattr(tts_provider, "synthesize_stream_bytes", getattr(tts_provider, "synthesize", None))
    async for sentence in stream_graph_sentences(query, asset_id, thread_id):
        # Strip markdown formatting symbols for clean speech synthesis
        clean_text = re.sub(r"[*#_\[\]\(\)]", "", sentence).strip()
        if clean_text and synth_fn:
            audio_bytes = await asyncio.to_thread(synth_fn, clean_text, voice)
            if audio_bytes and isinstance(audio_bytes, (bytes, bytearray)):
                yield bytes(audio_bytes)


@app.post("/agent/stream-text")
async def agent_text_endpoint(req: AgentQueryRequest):
    return StreamingResponse(
        stream_graph_sentences(req.query, req.asset_id, req.thread_id),
        media_type="text/plain",
    )


@app.post("/agent/stream-audio")
async def agent_audio_endpoint(req: AgentQueryRequest):
    """TTS: Streams live PCM audio chunks directly from the agent response."""
    return StreamingResponse(
        stream_audio_chunks(req.query, req.asset_id, req.voice, req.thread_id),
        media_type="audio/pcm",
    )


@app.post("/stt/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """STT: Transcribes an uploaded audio file using Qwen3-ASR."""
    suffix = f".{file.filename.split('.')[-1]}" if "." in file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = await asyncio.to_thread(stt_provider.transcribe_with_language, tmp_path)
    return result
