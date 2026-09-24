import os
import re
import asyncio
import tempfile
import json
import urllib.parse
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph.build import build_graph
from src.graph.nodes import AgentNodes
from src.mongo_store.mongo_db import DataBase
from src.tts.kokoro_provider import KokoroTTSProvider
from src.stt.qwen_asr_provider import QwenASRProvider
from src.tts.wav_utils import pcm_to_wav_bytes

load_dotenv()

app = FastAPI(title="LangGraph Voice Agent - Speech-to-Speech Streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
mongo = DataBase(db_name=os.getenv("DB_NAME", "task2"))
nodes = AgentNodes()
graph = build_graph(nodes)
tts_provider = KokoroTTSProvider()
stt_provider = QwenASRProvider()
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\n])\s+")


def clean_speech_text(text: str) -> str:
    text = re.sub(r"```(?:\w+)?\s*([\s\S]*?)```", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\d+(?:-\d+)?\]", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[*#_`]", "", text).strip()


class AgentQueryRequest(BaseModel):
    query: str
    asset_id: str = "22"
    session_id: str | None = None
    voice: str = "af_heart"
    thread_id: str = "default_session"


async def persist_session_history(asset_id: str, thread_id: str, query: str | None = None):
    try:
        session_id = thread_id or (query or "default_session")
        history = nodes.provider.export_history(asset_id, thread_id=thread_id)
        mongo.save_session_chat(session_id, asset_id, history)
    except Exception:
        pass


def load_session_history(asset_id: str, thread_id: str):
    session_id = thread_id or "default_session"
    history = mongo.get_session_chat(session_id, asset_id=asset_id)
    if history:
        nodes.provider.load_history_from_dicts(asset_id, history, thread_id=thread_id)


async def run_agent_events(query: str, asset_id: str, thread_id: str):
    """Yields structured events: token | action."""
    load_session_history(asset_id, thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    got_tokens, summary = False, ""

    async for mode, data in graph.astream(
        {"query": query, "asset_id": asset_id, "thread_id": thread_id},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "summarize" and isinstance(chunk.content, str) and chunk.content:
                got_tokens = True
                yield {"type": "token", "data": chunk.content}

        elif mode == "updates":
            for node, out in data.items():
                if not isinstance(out, dict):
                    continue
                if node == "summarize":
                    summary = out.get("summary", "")
                    if not got_tokens and summary:
                        yield {"type": "token", "data": summary}
                elif node == "act":
                    email = out.get("email_status")
                    report = out.get("report_path")
                    if email:
                        ok = not str(email).startswith("error")
                        yield {
                            "type": "action",
                            "kind": "email",
                            "ok": ok,
                            "to": email.removeprefix("sent to ") if ok else None,
                            "error": None if ok else email,
                            "subject": "Summary",
                            "preview": summary[:240],
                        }
                    if report:
                        ok = not str(report).startswith("error")
                        yield {
                            "type": "action",
                            "kind": "report",
                            "ok": ok,
                            "file": os.path.basename(report) if ok else None,
                            "error": None if ok else report,
                        }


async def ndjson_stream(query, asset_id, thread_id):
    try:
        async for evt in run_agent_events(query, asset_id, thread_id):
            yield json.dumps(evt) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"
    await persist_session_history(asset_id, thread_id)
    yield json.dumps({"type": "done"}) + "\n"


@app.get("/agent/history")
async def get_agent_history(
    thread_id: str | None = None,
    session_id: str | None = None,
    asset_id: str = "22",
):
    session_key = thread_id or session_id
    if not session_key:
        return {"messages": []}
    history = mongo.get_session_chat(session_key, asset_id=asset_id)
    return {"messages": history}


async def stream_graph_sentences(query: str, asset_id: str, thread_id: str):
    """Kept for the legacy audio endpoints: sentence-buffered text."""
    buf = ""
    async for evt in run_agent_events(query, asset_id, thread_id):
        if evt["type"] != "token":
            continue
        buf += evt["data"]
        parts = SENTENCE_SPLIT_REGEX.split(buf)
        for s in parts[:-1]:
            if s.strip():
                yield s.strip() + "\n\n"
        buf = parts[-1]
    if buf.strip():
        yield buf.strip() + "\n"


async def stream_audio_chunks(query: str, asset_id: str, voice: str, thread_id: str) -> AsyncGenerator[bytes, None]:
    """Converts buffered agent text chunks into PCM audio bytes via Kokoro TTS."""
    synth_fn = getattr(tts_provider, "synthesize_stream_bytes", getattr(tts_provider, "synthesize", None))
    async for sentence in stream_graph_sentences(query, asset_id, thread_id):
        clean_text = clean_speech_text(sentence)
        if clean_text and synth_fn:
            audio_bytes = await asyncio.to_thread(synth_fn, clean_text, voice)
            if audio_bytes and isinstance(audio_bytes, (bytes, bytearray)):
                yield bytes(audio_bytes)


@app.post("/agent/stream-text")
async def agent_text_endpoint(req: AgentQueryRequest):
    if req.session_id:
        req.thread_id = req.session_id
    return StreamingResponse(
        ndjson_stream(req.query, req.asset_id, req.thread_id),
        media_type="application/x-ndjson",
    )


@app.post("/agent/stream-audio")
async def agent_audio_endpoint(req: AgentQueryRequest):
    """Streams synthesized PCM audio directly from text input."""
    return StreamingResponse(
        stream_audio_chunks(req.query, req.asset_id, req.voice, req.thread_id),
        media_type="audio/pcm",
    )


@app.post("/stt/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Transcribes an uploaded audio file using Qwen3-ASR."""
    suffix = f".{file.filename.split('.')[-1]}" if "." in (file.filename or "") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcription_fn = getattr(stt_provider, "transcribe_with_language", getattr(stt_provider, "transcribe", None))
        result = await asyncio.to_thread(transcription_fn, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if isinstance(result, dict):
        return result
    return {"text": str(result)}


@app.post("/agent/speech-to-speech")
async def speech_to_speech_endpoint(
    file: UploadFile = File(...),
    asset_id: str = Form("22"),
    voice: str = Form("af_heart"),
    thread_id: str = Form("voice_session"),
):
    """End-to-End Voice Pipeline via HTTP multipart upload."""
    suffix = f".{file.filename.split('.')[-1]}" if "." in (file.filename or "") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcription = await asyncio.to_thread(stt_provider.transcribe, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    transcribed_text = transcription.get("text", "") if isinstance(transcription, dict) else str(transcription or "")
    transcribed_text = transcribed_text.strip()
    if not transcribed_text:
        raise HTTPException(status_code=400, detail="No speech detected in audio file.")

    safe_header_text = urllib.parse.quote(transcribed_text)
    return StreamingResponse(
        stream_audio_chunks(
            query=transcribed_text,
            asset_id=asset_id,
            voice=voice,
            thread_id=thread_id,
        ),
        media_type="audio/pcm",
        headers={"X-Transcribed-Text": safe_header_text},
    )


@app.websocket("/ws/converse")
async def converse_ws(ws: WebSocket):
    import base64
    await ws.accept()
    synth_fn = getattr(tts_provider, "synthesize_stream_bytes", None)
    audio_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

    async def tts_worker():
        while True:
            item = await audio_queue.get()
            if item is None:
                audio_queue.task_done()
                return
            sentence, voice = item
            try:
                if synth_fn:
                    pcm = await asyncio.to_thread(synth_fn, sentence, voice)
                    if pcm:
                        await ws.send_json({
                            "type": "audio_chunk",
                            "data": base64.b64encode(pcm_to_wav_bytes(pcm)).decode(),
                        })
            except Exception as exc:
                await ws.send_json({"type": "error", "data": f"TTS error: {exc}"})
            finally:
                audio_queue.task_done()

    tts_task = asyncio.create_task(tts_worker())

    try:
        while True:
            msg = await ws.receive_json()
            voice = msg.get("voice", "af_heart")
            asset_id = msg.get("asset_id", "22")
            thread_id = msg.get("thread_id") or msg.get("session_id") or "default_session"

            if msg.get("type") == "audio":
                wav = base64.b64decode(msg["data"])
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav)
                    path = tmp.name
                try:
                    res = await asyncio.to_thread(stt_provider.transcribe, path)
                finally:
                    os.remove(path)
                query = (res.get("text", "") if isinstance(res, dict) else str(res or "")).strip()
                await ws.send_json({"type": "transcript", "data": query})
                if not query:
                    await ws.send_json({"type": "done", "empty": True})
                    continue
            else:
                query = msg.get("data", "").strip()

            buf = ""
            try:
                async for evt in run_agent_events(query, asset_id, thread_id):
                    if evt["type"] == "token":
                        await ws.send_json({"type": "text_chunk", "data": evt["data"]})
                        buf += evt["data"]
                        parts = SENTENCE_SPLIT_REGEX.split(buf)
                        for s in parts[:-1]:
                            clean = clean_speech_text(s)
                            if clean:
                                await audio_queue.put((clean, voice))
                        buf = parts[-1]
                    else:
                        await ws.send_json(evt)
                if buf.strip():
                    clean = clean_speech_text(buf)
                    if clean:
                        await audio_queue.put((clean, voice))
            except Exception as e:
                await ws.send_json({"type": "error", "data": str(e)})
            await audio_queue.join()
            await persist_session_history(asset_id, thread_id)
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    finally:
        await audio_queue.put(None)
        await tts_task


@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail=f"index.html not found in {FRONTEND_DIR}")
    return FileResponse(index_file)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

