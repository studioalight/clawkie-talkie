// Self-test: run the real OpenClawInferSttSession path (sendAudio -> signalAudioDone -> onDone)
import { OpenClawInferSttSession } from '../src/inferSttSession.js';
import { readFileSync } from 'node:fs';
import { pcm16ToWavBuffer } from '../src/audio.js';

// Generate 3s of 16kHz mono PCM16 tone (440Hz) like a real mic chunk stream (20ms chunks)
const sampleRate = 16000;
const durationMs = 3000;
const chunks: Buffer[] = [];
const samplesPerChunk = (sampleRate * 20) / 1000; // 320 samples per 20ms chunk
for (let c = 0; c < durationMs / 20; c++) {
  const buf = Buffer.alloc(samplesPerChunk * 2);
  for (let i = 0; i < samplesPerChunk; i++) {
    const t = ((c * samplesPerChunk) + i) / sampleRate;
    // tone + enough amplitude so VAD treats as speech-ish
    buf.writeInt16LE(Math.round(8000 * Math.sin(2 * Math.PI * 200 * t)), i * 2);
  }
  chunks.push(buf);
}

const session = new OpenClawInferSttSession(
  { sampleRate, enablePhraseChunks: false },
  {
    onReady: () => console.log('[test] onReady'),
    onPartial: (t, f) => console.log(`[test] onPartial "${t}" final=${f}`),
    onDone: (text) => { console.log(`[test] onDone: "${text.slice(0, 120)}"`); process.exit(0); },
    onError: (m) => { console.log(`[test] onError: ${m}`); process.exit(1); },
    onClosed: () => console.log('[test] onClosed'),
  },
);

console.log('[test] session created, sending ', chunks.length, 'chunks');
for (const c of chunks) session.sendAudio(new Uint8Array(c));
setTimeout(() => { session.signalAudioDone(); }, 100);

// Safety net: if nothing happens in 150s, fail
setTimeout(() => { console.log('[test] TIMEOUT after 150s — turn still wedged'); process.exit(2); }, 150000);