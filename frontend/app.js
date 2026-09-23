// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/converse";
const ASSET_ID = "22";
const THREAD_ID = "default_session";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const chatEl = document.getElementById("chat");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const voiceSelect = document.getElementById("voiceSelect");
const statusDot = document.getElementById("statusDot");
const recIndicator = document.getElementById("recIndicator");

// ---------------------------------------------------------------------------
// WAV recorder (raw PCM capture -> real .wav bytes, no ffmpeg needed)
// ---------------------------------------------------------------------------
class WavRecorder {
  constructor(sampleRate = 16000) {
    this.sampleRate = sampleRate;
    this.chunks = [];
    this.ctx = null;
    this.processor = null;
    this.source = null;
    this.stream = null;
  }

  async start() {
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: this.sampleRate, echoCancellation: true, noiseSuppression: true },
    });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: this.sampleRate });
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
    };

    this.source.connect(this.processor);
    this.processor.connect(this.ctx.destination);
  }

  stop() {
    if (this.processor) this.processor.disconnect();
    if (this.source) this.source.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close();

    const merged = this._mergeChunks(this.chunks);
    return this._encodeWav(merged, this.sampleRate);
  }

  _mergeChunks(chunks) {
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
      result.set(c, offset);
      offset += c.length;
    }
    return result;
  }

  _encodeWav(float32, sampleRate) {
    const numFrames = float32.length;
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample; // mono
    const byteRate = sampleRate * blockAlign;
    const dataSize = numFrames * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);       // PCM chunk size
    view.setUint16(20, 1, true);        // audio format = PCM
    view.setUint16(22, 1, true);        // channels = mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);       // bits per sample
    writeStr(36, "data");
    view.setUint32(40, dataSize, true);

    // Float32 [-1,1] -> Int16 PCM
    let offset = 44;
    for (let i = 0; i < numFrames; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([buffer], { type: "audio/wav" });
  }
}

// ---------------------------------------------------------------------------
// Sequential audio playback queue (plays TTS chunks in generation order)
// ---------------------------------------------------------------------------
class AudioQueue {
  constructor() {
    this.queue = [];
    this.playing = false;
  }

  push(blobUrl) {
    this.queue.push(blobUrl);
    if (!this.playing) this._playNext();
  }

  _playNext() {
    if (this.queue.length === 0) {
      this.playing = false;
      setStatusSpeaking(false);
      return;
    }
    this.playing = true;
    setStatusSpeaking(true);
    const url = this.queue.shift();
    const audio = new Audio(url);
    audio.onended = () => {
      URL.revokeObjectURL(url);
      this._playNext();
    };
    audio.onerror = () => this._playNext();
    audio.play().catch(() => this._playNext());
  }
}

const audioQueue = new AudioQueue();

// ---------------------------------------------------------------------------
// Chat rendering helpers
// ---------------------------------------------------------------------------
function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function appendAudioIcon(msgEl) {
  if (msgEl.querySelector(".audio-icon")) return;
  const icon = document.createElement("div");
  icon.className = "audio-icon";
  icon.innerHTML = `<span class="bars"><span></span><span></span><span></span></span> voice`;
  msgEl.appendChild(icon);
}

function setStatusSpeaking(isSpeaking) {
  statusDot.classList.toggle("connected", isSpeaking || wsIsOpen());
}

function wsIsOpen() {
  return ws && ws.readyState === WebSocket.OPEN;
}

// ---------------------------------------------------------------------------
// WebSocket lifecycle
// ---------------------------------------------------------------------------
let ws;
let currentBotBubble = null;

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    statusDot.classList.add("connected");
  };

  ws.onclose = () => {
    statusDot.classList.remove("connected");
    setTimeout(connect, 1500); // auto-reconnect
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    switch (msg.type) {
      case "transcript":
        addMessage(msg.data, "msg transcript");
        break;

      case "text_chunk":
        if (!currentBotBubble) {
          currentBotBubble = addMessage("", "bot streaming");
        }
        currentBotBubble.textContent += (currentBotBubble.textContent ? " " : "") + msg.data;
        chatEl.scrollTop = chatEl.scrollHeight;
        break;

      case "audio_chunk": {
        const bytes = atob(msg.data);
        const arr = new Uint8Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
        const blob = new Blob([arr], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        audioQueue.push(url);
        if (currentBotBubble) appendAudioIcon(currentBotBubble);
        break;
      }

      case "done":
        if (currentBotBubble) currentBotBubble.classList.remove("streaming");
        currentBotBubble = null;
        break;

      default:
        break;
    }
  };
}

// ---------------------------------------------------------------------------
// Send helpers
// ---------------------------------------------------------------------------
function sendText(text) {
  if (!text.trim() || !wsIsOpen()) return;
  addMessage(text, "user");
  ws.send(JSON.stringify({
    type: "text",
    data: text,
    asset_id: ASSET_ID,
    thread_id: THREAD_ID,
    voice: voiceSelect.value,
  }));
  textInput.value = "";
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function sendAudioBlob(blob) {
  if (!wsIsOpen()) return;
  const b64 = await blobToBase64(blob);
  ws.send(JSON.stringify({
    type: "audio",
    data: b64,
    asset_id: ASSET_ID,
    thread_id: THREAD_ID,
    voice: voiceSelect.value,
  }));
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
sendBtn.addEventListener("click", () => sendText(textInput.value));
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendText(textInput.value);
});

let recorder = null;
let isRecording = false;

async function startRecording() {
  if (isRecording) return;
  try {
    recorder = new WavRecorder(16000);
    await recorder.start();
    isRecording = true;
    micBtn.classList.add("recording");
    recIndicator.classList.remove("hidden");
  } catch (err) {
    console.error("Mic access failed:", err);
  }
}

async function stopRecording() {
  if (!isRecording || !recorder) return;
  isRecording = false;
  micBtn.classList.remove("recording");
  recIndicator.classList.add("hidden");
  const wavBlob = recorder.stop();
  await sendAudioBlob(wavBlob);
}

// Push-to-talk: hold mouse/touch to record
micBtn.addEventListener("mousedown", startRecording);
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", () => { if (isRecording) stopRecording(); });
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
micBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
connect();