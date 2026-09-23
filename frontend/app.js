const $ = (id) => document.getElementById(id);
const chat = $('chat'), messages = $('messages'), empty = $('empty'), input = $('user-input');
const statusText = $('status-text'), statusDot = $('status-dot');
const cfg = () => ({
  asset_id: $('asset-id-input').value.trim() || '22',
  thread_id: $('thread-id-input').value.trim() || 'default_session',
  voice: $('voice-select').value,
});
const setStatus = (t, cls = '') => { statusText.textContent = t; statusDot.className = 'status-dot ' + cls; };
const scrollDown = () => (chat.scrollTop = chat.scrollHeight);

async function loadSessionHistory() {
  const { thread_id, asset_id } = cfg();
  if (!thread_id) return;
  try {
    const res = await fetch(`/agent/history?thread_id=${encodeURIComponent(thread_id)}&asset_id=${encodeURIComponent(asset_id)}`);
    if (!res.ok) return;
    const data = await res.json();
    const messagesData = Array.isArray(data.messages) ? data.messages : [];
    if (!messagesData.length) return;

    messages.innerHTML = '';
    empty.classList.add('hidden');

    for (const item of messagesData) {
      const content = (item.content || '').trim();
      if (!content) continue;
      if (item.role === 'human' && content.startsWith('Summarize the answer to the request below')) continue;
      if (item.role === 'human') {
        addUser(content);
      } else if (item.role === 'ai') {
        const turn = newAssistantTurn();
        turn.token(cleanGeneratedText(content));
        turn.finish();
      }
    }
  } catch (err) {
    console.warn('Could not restore session history:', err);
  }
}

class WavAudioRecorder {
  constructor(rate = 16000) { this.rate = rate; this.pcmChunks = []; this.isRecording = false; this.onFrame = null; }
  async start() {
    this.pcmChunks = [];
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    this.src = this.ctx.createMediaStreamSource(this.stream);
    this.proc = this.ctx.createScriptProcessor(4096, 1, 1);
    this.proc.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const data = e.inputBuffer.getChannelData(0);
      let sum = 0; for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      this.pcmChunks.push(this.downsample(data, this.ctx.sampleRate, this.rate));
      if (this.onFrame) this.onFrame(Math.sqrt(sum / data.length));
    };
    this.src.connect(this.proc); this.proc.connect(this.ctx.destination);
    this.isRecording = true;
  }
  downsample(buf, from, to) {
    if (from === to) return new Float32Array(buf);
    const ratio = from / to, out = new Float32Array(Math.round(buf.length / ratio));
    let o = 0, b = 0;
    while (o < out.length) {
      const next = Math.round((o + 1) * ratio); let acc = 0, n = 0;
      for (let i = b; i < next && i < buf.length; i++) { acc += buf[i]; n++; }
      out[o++] = n ? acc / n : 0; b = next;
    }
    return out;
  }
  flush() {
    const total = this.pcmChunks.reduce((a, c) => a + c.length, 0), m = new Float32Array(total);
    let off = 0; for (const c of this.pcmChunks) { m.set(c, off); off += c.length; }
    this.pcmChunks = [];
    return this.encodeWAV(m, this.rate);
  }
  async stop() {
    this.isRecording = false;
    this.src?.disconnect(); this.proc?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close();
    return this.flush();
  }
  encodeWAV(s, rate) {
    const buf = new ArrayBuffer(44 + s.length * 2), v = new DataView(buf);
    const w = (o, str) => [...str].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    w(0, 'RIFF'); v.setUint32(4, 36 + s.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, s.length * 2, true);
    let i = 44; for (const x of s) { const c = Math.max(-1, Math.min(1, x)); v.setInt16(i, c < 0 ? c * 0x8000 : c * 0x7fff, true); i += 2; }
    return new Blob([v], { type: 'audio/wav' });
  }
}

class Typewriter {
  constructor(render) { this.q = ''; this.raw = ''; this.render = render; this.timer = null; this.onIdle = null; }
  push(t) { this.q += t; if (!this.timer) this.timer = setInterval(() => this.tick(), 16); }
  tick() {
    if (!this.q) { clearInterval(this.timer); this.timer = null; this.onIdle?.(); return; }
    const n = Math.max(1, Math.ceil(this.q.length / 40));
    this.raw += this.q.slice(0, n); this.q = this.q.slice(n);
    this.render(this.raw);
  }
  whenDone(cb) { this.timer ? (this.onIdle = cb) : cb(); }
}

function cleanGeneratedText(text) {
  return String(text || '')
    .replace(/```(?:\w+)?\s*([\s\S]*?)```/g, '$1')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[-*_]{3,}\s*$/gm, '')
    .replace(/[ \t]+\n/g, '\n')
    .trim();
}
const md = (t) => DOMPurify.sanitize(marked.parse(cleanGeneratedText(t), { breaks: true }));
function addUser(text) {
  empty.classList.add('hidden');
  const d = document.createElement('div'); d.className = 'msg user';
  d.innerHTML = '<div class="bubble"></div>'; d.firstChild.textContent = text;
  messages.appendChild(d); scrollDown(); return d.firstChild;
}
function addSystem(text) {
  const d = document.createElement('div'); d.className = 'msg system'; d.textContent = text;
  messages.appendChild(d); scrollDown();
}
function newAssistantTurn() {
  empty.classList.add('hidden');
  const el = document.createElement('div'); el.className = 'msg assistant';
  const bubble = document.createElement('div'); bubble.className = 'bubble cursor';
  el.appendChild(bubble); messages.appendChild(el);
  const tw = new Typewriter((raw) => { bubble.innerHTML = md(raw); scrollDown(); });
  return {
    el, bubble, tw,
    token: (t) => tw.push(t),
    action: (evt) => tw.whenDone(() => { el.appendChild(actionCard(evt)); scrollDown(); }),
    finish: () => tw.whenDone(() => bubble.classList.remove('cursor')),
    error: (m) => { bubble.classList.remove('cursor'); bubble.textContent = `Something went wrong: ${m}`; },
  };
}
function actionCard(e) {
  const c = document.createElement('div'); c.className = 'action-card' + (e.ok ? '' : ' fail');
  const email = e.kind === 'email';
  const title = email ? (e.ok ? 'Email sent' : 'Email failed') : (e.ok ? 'Report saved' : 'Report failed');
  const sub = e.ok ? (email ? `To ${e.to}` + (e.preview ? ` · ${e.preview}` : '') : e.file) : e.error;
  c.innerHTML = `<div class="ico">${email ? '✉️' : '📄'}</div>
    <div class="meta"><div class="t"></div><div class="s"></div></div>
    <span class="pill">${e.ok ? 'Done' : 'Failed'}</span>`;
  c.querySelector('.t').textContent = title; c.querySelector('.s').textContent = sub || '';
  return c;
}

async function sendText(query) {
  addUser(query);
  const turn = newAssistantTurn();
  setStatus('Thinking…', 'busy');
  try {
    const res = await fetch('/agent/stream-text', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ...cfg() }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const l of lines) if (l.trim()) handleEvent(JSON.parse(l), turn);
    }
  } catch (e) { turn.error(e.message); } finally { turn.finish(); setStatus('Ready'); }
}
function handleEvent(evt, turn) {
  if (evt.type === 'token') turn.token(evt.data);
  else if (evt.type === 'action') turn.action(evt);
  else if (evt.type === 'error') turn.error(evt.data);
}

function handleSend() {
  const q = input.value.trim(); if (!q) return;
  input.value = ''; input.style.height = 'auto'; sendText(q);
}
$('send-btn').onclick = handleSend;
input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } });
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 180) + 'px'; });
document.querySelectorAll('.chip').forEach((c) => (c.onclick = () => { input.value = c.textContent; input.focus(); }));
$('settings-btn').onclick = () => $('settings-panel').classList.toggle('hidden');
$('new-chat-btn').onclick = () => {
  $('thread-id-input').value = 'session_' + Math.random().toString(36).slice(2, 9);
  messages.innerHTML = ''; empty.classList.remove('hidden');
  loadSessionHistory();
};

$('thread-id-input').addEventListener('change', loadSessionHistory);
window.addEventListener('load', loadSessionHistory);

const dictRec = new WavAudioRecorder(16000);
$('dictate-btn').onclick = async () => {
  const btn = $('dictate-btn');
  if (!dictRec.isRecording) {
    try { await dictRec.start(); } catch (e) { return addSystem('Microphone error: ' + e.message); }
    btn.classList.add('recording'); setStatus('Recording…', 'rec');
    $('hint').textContent = 'Recording… click the mic again to stop';
  } else {
    btn.classList.remove('recording'); setStatus('Transcribing…', 'busy');
    const wav = await dictRec.stop();
    try {
      const fd = new FormData(); fd.append('file', wav, 'rec.wav');
      const r = await fetch('/stt/transcribe', { method: 'POST', body: fd });
      const data = await r.json();
      const text = (data.text || '').trim();
      if (text) { input.value = (input.value ? input.value + ' ' : '') + text; input.dispatchEvent(new Event('input')); input.focus(); }
      else addSystem('No speech recognized.');
    } catch (e) { addSystem('STT error: ' + e.message); }
    setStatus('Ready'); $('hint').textContent = 'Enter to send · Shift+Enter for a new line';
  }
};

let ws = null, wsReady = null, activeTurn = null;
const audioQueue = []; let playing = false, serverDone = true;

function ensureWs() {
  if (ws && ws.readyState === WebSocket.OPEN) return Promise.resolve();
  if (wsReady) return wsReady;
  wsReady = new Promise((res, rej) => {
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/converse`);
    ws.onopen = () => { wsReady = null; res(); };
    ws.onerror = () => { wsReady = null; rej(new Error('WebSocket error')); };
    ws.onclose = () => { wsReady = null; };
    ws.onmessage = (e) => onWsMessage(JSON.parse(e.data));
  });
  return wsReady;
}
function onWsMessage(m) {
  if (m.type === 'transcript') return voice.onTranscript(m.data);
  if (m.type === 'audio_chunk') return queueAudio(m.data);
  if (m.type === 'done') { serverDone = true; activeTurn?.finish(); return voice.maybeResume(); }
  if (activeTurn) handleEvent(m.type === 'text_chunk' ? { type: 'token', data: m.data } : m, activeTurn);
}
function queueAudio(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  audioQueue.push(URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })));
  voice.setState('speaking');
  if (!playing) playNext();
}
function playNext() {
  if (!audioQueue.length) { playing = false; voice.maybeResume(); return; }
  playing = true;
  const url = audioQueue.shift(), a = new Audio(url);
  const next = () => { URL.revokeObjectURL(url); playNext(); };
  a.onended = next; a.onerror = next; a.play().catch(next);
}

const SPEECH_RMS = 0.06, SILENCE_MS = 1100, MIN_SPEECH_MS = 300;
const voice = {
  state: 'listening', muted: false, speaking: false, lastVoice: 0, startedAt: 0, rec: null, continueRequested: false, speechSamples: 0,
  async open() {
    try { await ensureWs(); this.rec = new WavAudioRecorder(16000); this.rec.onFrame = (r) => this.onFrame(r); await this.rec.start(); }
    catch (e) { return addSystem('Voice mode error: ' + e.message); }
    $('voice-overlay').classList.remove('hidden');
    this.muted = false; this.continueRequested = false; this.speechSamples = 0; $('voice-mute').classList.remove('off'); $('voice-transcript').textContent = '';
    this.setState('listening');
  },
  async close() {
    $('voice-overlay').classList.add('hidden');
    audioQueue.length = 0; playing = false;
    await this.rec?.stop(); this.rec = null; this.speaking = false; this.continueRequested = false; this.speechSamples = 0;
  },
  setState(s) {
    this.state = s; const orb = $('orb');
    const isMicMutedOnly = this.muted && s === 'listening';
    orb.className = 'orb ' + (isMicMutedOnly ? 'muted' : s);
    $('voice-status').textContent = isMicMutedOnly ? 'Mic muted' : { listening: 'Listening…', thinking: 'Thinking…', speaking: 'Speaking…' }[s] || 'Listening…';
    const level = s === 'listening' ? 0.12 : s === 'thinking' ? 0.22 : s === 'speaking' ? 0.32 : 0;
    orb.style.setProperty('--level', level);
  },
  onFrame(rms) {
    if (this.state !== 'listening') return;
    if (this.muted) {
      this.rec?.pcmChunks && (this.rec.pcmChunks = []);
      this.speechSamples = 0;
      return;
    }
    $('orb').style.setProperty('--level', Math.min(rms * 8, 0.35));
    const now = performance.now();
    if (rms > SPEECH_RMS) {
      this.speechSamples += 1;
      if (this.speechSamples >= 3 && !this.speaking) {
        this.speaking = true;
        this.startedAt = now;
      }
      this.lastVoice = now;
    } else {
      this.speechSamples = Math.max(0, this.speechSamples - 1);
      if (!this.speaking) {
        if (this.rec.pcmChunks.length > 3) this.rec.pcmChunks.shift();
      }
    }

    if (this.speaking && now - this.lastVoice > SILENCE_MS) {
      this.speaking = false;
      if (now - this.startedAt - SILENCE_MS < MIN_SPEECH_MS) {
        this.rec.pcmChunks = [];
        this.speechSamples = 0;
        return;
      }
      this.sendUtterance();
    }
  },
  sendUtterance() {
    if (!this.rec || !ws || ws.readyState !== WebSocket.OPEN) return;
    const wav = this.rec.flush(); this.setState('thinking'); serverDone = false;
    this.speechSamples = 0;
    const r = new FileReader();
    r.onloadend = () => ws.send(JSON.stringify({ type: 'audio', data: r.result.split(',')[1], ...cfg() }));
    r.readAsDataURL(wav);
  },
  onTranscript(text) {
    if (!text) return;
    const clean = text.trim();
    $('voice-transcript').textContent = clean;
    if (/^continue\b/i.test(clean)) {
      this.continueRequested = true;
      this.setState('listening');
      $('voice-transcript').textContent = 'Continuing…';
      return;
    }
    addUser(clean); activeTurn = newAssistantTurn();
  },
  maybeResume() {
    if (this.continueRequested && !this.muted && !$('voice-overlay').classList.contains('hidden')) {
      this.continueRequested = false;
      this.rec?.pcmChunks && (this.rec.pcmChunks = []);
      this.setState('listening');
      return;
    }
    if (this.state !== 'listening' && serverDone && !playing && !audioQueue.length && !$('voice-overlay').classList.contains('hidden')) {
      this.rec?.pcmChunks && (this.rec.pcmChunks = []);
      this.setState('listening');
    }
  },
};
$('voice-btn').onclick = () => voice.open();
$('voice-close').onclick = () => voice.close();
$('voice-mute').onclick = () => {
  voice.muted = !voice.muted; $('voice-mute').classList.toggle('off', voice.muted); if (voice.state === 'listening') voice.setState('listening');
};
