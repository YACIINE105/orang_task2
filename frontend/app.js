/**
 * Unified Frontend Agent Controller
 * Supports: Chat Only, Speech-to-Speech (WebSocket), and Audio-to-Text (STT)
 */

class WavAudioRecorder {
  constructor(targetSampleRate = 16000) {
    this.targetSampleRate = targetSampleRate;
    this.audioContext = null;
    this.mediaStream = null;
    this.processor = null;
    this.input = null;
    this.pcmChunks = [];
    this.isRecording = false;
  }

  async start() {
    this.pcmChunks = [];
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.input = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const inputBuffer = e.inputBuffer.getChannelData(0);
      const downsampled = this.downsample(inputBuffer, this.audioContext.sampleRate, this.targetSampleRate);
      this.pcmChunks.push(downsampled);
    };

    this.input.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
    this.isRecording = true;
  }

  downsample(buffer, fromRate, toRate) {
    if (fromRate === toRate) return new Float32Array(buffer);
    const ratio = fromRate / toRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  async stop() {
    this.isRecording = false;
    if (this.processor && this.input) {
      this.input.disconnect();
      this.processor.disconnect();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
    }
    if (this.audioContext) {
      await this.audioContext.close();
    }

    const totalLength = this.pcmChunks.reduce((acc, curr) => acc + curr.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.pcmChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return this.encodeWAV(merged, this.targetSampleRate);
  }

  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, 'WAVE');
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    this.writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let index = 44;
    for (let i = 0; i < samples.length; i++, index += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}

// UI Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const cancelRecBtn = document.getElementById('cancel-rec-btn');
const recordingBar = document.getElementById('recording-bar');
const recordingTimer = document.getElementById('recording-timer');
const recordingModeLabel = document.getElementById('recording-mode-label');
const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('status-text');
const modeTabs = document.querySelectorAll('.mode-btn');
const modeHint = document.getElementById('mode-hint');
const toggleSettingsBtn = document.getElementById('toggle-settings-btn');
const settingsPanel = document.getElementById('settings-panel');
const voiceSelect = document.getElementById('voice-select');
const assetIdInput = document.getElementById('asset-id-input');
const threadIdInput = document.getElementById('thread-id-input');
const newChatBtn = document.getElementById('new-chat-btn');

// State
let currentMode = 'chat'; // 'chat' | 's2s' | 'stt'
let recorder = new WavAudioRecorder(16000);
let timerInterval = null;
let recordSeconds = 0;
let ws = null;
const audioQueue = [];
let isPlayingAudio = false;

// Mode Descriptions
const modeHints = {
  chat: 'Text Chat mode: streams text responses from the LangGraph agent.',
  s2s: 'Speech-to-Speech mode: click the mic to speak and hear live audio responses.',
  stt: 'Audio to Text mode: record voice to transcribe speech directly into text.'
};

// Initialize Mode Tabs
modeTabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    modeTabs.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    modeHint.textContent = modeHints[currentMode];
    if (recorder.isRecording) cancelRecording();
  });
});

// Settings Drawer Toggle
toggleSettingsBtn.addEventListener('click', () => {
  settingsPanel.classList.toggle('hidden');
});

// New Chat Session
newChatBtn.addEventListener('click', () => {
  threadIdInput.value = 'session_' + Math.random().toString(36).substring(2, 9);
  chatMessages.innerHTML = '';
  appendMessage('system', `Started new session: ${threadIdInput.value}`);
});

// Auto-expand textarea
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
});

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);
micBtn.addEventListener('click', toggleRecording);
cancelRecBtn.addEventListener('click', cancelRecording);

// Message UI Helpers
function appendMessage(role, text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  msgDiv.appendChild(bubble);
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

function setStatus(text, dotClass = '') {
  statusText.textContent = text;
  statusDot.className = `status-dot ${dotClass}`.trim();
}

// 1. Text Chat Mode (POST /agent/stream-text)
async function handleTextChat(query) {
  setStatus('Thinking...', 'busy');
  const assistantBubble = appendMessage('assistant', '');

  try {
    const response = await fetch('/agent/stream-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        asset_id: assetIdInput.value.trim() || '22',
        voice: voiceSelect.value,
        thread_id: threadIdInput.value.trim() || 'default_session'
      })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let accumulatedText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      accumulatedText += decoder.decode(value, { stream: true });
      assistantBubble.textContent = accumulatedText;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  } catch (err) {
    assistantBubble.textContent = `Error streaming response: ${err.message}`;
  } finally {
    setStatus('Ready');
  }
}

// 2. Speech-to-Speech Mode (WebSocket /ws/converse)
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/converse`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => setStatus('Ready');
  ws.onerror = () => setStatus('WebSocket Error', 'busy');
}

connectWebSocket();

async function handleSpeechToSpeech(wavBlob) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWebSocket();
    await new Promise((r) => setTimeout(r, 500));
  }

  setStatus('Processing voice...', 'busy');
  const userBubble = appendMessage('user', '🎤 Voice input...');
  const assistantBubble = appendMessage('assistant', '');

  const reader = new FileReader();
  reader.onloadend = () => {
    const base64Audio = reader.result.split(',')[1];
    ws.send(JSON.stringify({
      type: 'audio',
      data: base64Audio,
      asset_id: assetIdInput.value.trim() || '22',
      voice: voiceSelect.value,
      thread_id: threadIdInput.value.trim() || 'default_session'
    }));
  };
  reader.readAsDataURL(wavBlob);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'transcript') {
      userBubble.textContent = `🗣️ "${msg.data}"`;
    } else if (msg.type === 'text_chunk') {
      assistantBubble.textContent += msg.data;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    } else if (msg.type === 'audio_chunk') {
      queueAudioWav(msg.data);
    } else if (msg.type === 'done') {
      setStatus('Ready');
    }
  };
}

// Audio Sequential Playback
function queueAudioWav(base64Wav) {
  const binary = atob(base64Wav);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  audioQueue.push(url);
  if (!isPlayingAudio) playNextAudio();
}

function playNextAudio() {
  if (audioQueue.length === 0) {
    isPlayingAudio = false;
    return;
  }
  isPlayingAudio = true;
  const audioUrl = audioQueue.shift();
  const audio = new Audio(audioUrl);
  audio.onended = () => {
    URL.revokeObjectURL(audioUrl);
    playNextAudio();
  };
  audio.onerror = () => {
    URL.revokeObjectURL(audioUrl);
    playNextAudio();
  };
  audio.play().catch(() => playNextAudio());
}

// 3. Audio to Text Dictation (POST /stt/transcribe)
async function handleDictation(wavBlob) {
  setStatus('Transcribing audio...', 'busy');
  const formData = new FormData();
  formData.append('file', wavBlob, 'recording.wav');

  try {
    const res = await fetch('/stt/transcribe', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const transcribed = typeof data === 'string' ? data : (data.text || '');

    if (transcribed) {
      userInput.value = (userInput.value ? userInput.value + ' ' : '') + transcribed;
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
      userInput.focus();
    } else {
      appendMessage('system', 'No speech recognized from audio recording.');
    }
  } catch (err) {
    appendMessage('system', `STT Error: ${err.message}`);
  } finally {
    setStatus('Ready');
  }
}

// Send trigger
function handleSend() {
  const query = userInput.value.trim();
  if (!query) return;

  userInput.value = '';
  userInput.style.height = 'auto';
  appendMessage('user', query);

  if (currentMode === 's2s') {
    // S2S text submission also routes through WebSocket
    const assistantBubble = appendMessage('assistant', '');
    ws.send(JSON.stringify({
      type: 'text',
      data: query,
      asset_id: assetIdInput.value.trim() || '22',
      voice: voiceSelect.value,
      thread_id: threadIdInput.value.trim() || 'default_session'
    }));

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'text_chunk') {
        assistantBubble.textContent += msg.data;
        chatMessages.scrollTop = chatMessages.scrollHeight;
      } else if (msg.type === 'audio_chunk') {
        queueAudioWav(msg.data);
      } else if (msg.type === 'done') {
        setStatus('Ready');
      }
    };
  } else {
    // Default chat
    handleTextChat(query);
  }
}

// Recording Controls
async function toggleRecording() {
  if (recorder.isRecording) {
    stopRecordingAndProcess();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    await recorder.start();
    micBtn.classList.add('recording');
    recordingBar.classList.remove('hidden');
    recordingModeLabel.textContent = currentMode === 'stt' ? 'Dictating speech...' : 'Speaking to agent...';
    setStatus('Recording...', 'recording');

    recordSeconds = 0;
    recordingTimer.textContent = '00:00';
    timerInterval = setInterval(() => {
      recordSeconds++;
      const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
      const secs = String(recordSeconds % 60).padStart(2, '0');
      recordingTimer.textContent = `${mins}:${secs}`;
    }, 1000);
  } catch (err) {
    appendMessage('system', `Microphone access error: ${err.message}`);
    cancelRecording();
  }
}

async function stopRecordingAndProcess() {
  clearInterval(timerInterval);
  micBtn.classList.remove('recording');
  recordingBar.classList.add('hidden');
  setStatus('Processing...', 'busy');

  const wavBlob = await recorder.stop();

  if (currentMode === 'stt') {
    await handleDictation(wavBlob);
  } else {
    await handleSpeechToSpeech(wavBlob);
  }
}

function cancelRecording() {
  clearInterval(timerInterval);
  if (recorder.isRecording) {
    recorder.stop();
  }
  micBtn.classList.remove('recording');
  recordingBar.classList.add('hidden');
  setStatus('Ready');
}
