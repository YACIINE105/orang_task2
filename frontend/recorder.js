class WavRecorder {
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
    this.ctx = new AudioContext({ sampleRate: 16000 });
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.processor.onaudioprocess = e => this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(this.processor);
    this.processor.connect(this.ctx.destination);
  }
  stop() {
    this.processor.disconnect();
    this.stream.getTracks().forEach(t => t.stop());
    return this.encodeWav(this.mergeChunks());
  }
  mergeChunks() { /* flatten Float32Array[] into one */ }
  encodeWav(float32) { /* write 44-byte WAV header + 16-bit PCM, return Blob */ }
}

