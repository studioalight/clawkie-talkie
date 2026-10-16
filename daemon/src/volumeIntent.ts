/**
 * volumeIntent — decision-model volume-request detector for voice turns (field
 * exploration: the daemon detects "turn it up"/"max volume"/"set it to ..." on
 * voice transcripts BEFORE the chat turn, applies the gain in-process, and
 * returns a short spoken ack without entering an agent turn.
 *
 * Exploration note (2026-10-16): the daemon is not an agent context, so the
 * typed `decision_evaluate` is not directly reachable from this process; the
 * current call path is the same `openclaw agent` CLI used for chat turns, with
 * a boolean-style prompt, until an agent-side typed route lands. The detector
 * interface (`evaluateVolumeIntent`) is the seam to swap to the typed call.
 * Rate-limited + cached 6s (volume requests arrive in bursts on repeat PTT).
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export interface VolumeIntent {
  requested: boolean;      // probability >= threshold
  absoluteDb: number | null; // "set volume to N/max"
  adjustDb: number | null;   // "louder"/"quieter"/"a bit"
  probability: number;
}

const CACHE_WINDOW_MS = 6000;
let lastCallAt = 0;
let lastTranscript = '';
let lastResult: VolumeIntent | null = null;

const THRESHOLD = 0.85;
const MODEL = 'ollama-cloud/glm-5.3-flash';  // the voice agent's chat model; a fast one

const BOOLEAN_PROMPT = (t: string) => [
  `You are a voice-agent intent classifier. Decide whether the following is a`,
  `request to change playback volume. Answer ONLY "yes" or "no" and nothing`,
  `else (no punctuation, no explanation). Examples of yes: "turn it up",
  "louder", "quieter please", "set volume to max", "drop it by six", "volume
  down", "a bit quieter", "max volume", "raise the volume". Yes also includes
  the same request in another language. Any command that merely mentions
  playback (e.g. "play") without a change request is no. Transcript:
  "${t.replace(/"/g, "'").slice(0, 220)}"`,
].join(' ');

function normalizeNumeric(text: string): number | null {
  const m = text.match(/\b(\d{1,2})\b/i);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  if (isNaN(n)) return null;
  return n;
}

function parseRelativeAmount(t: string): number | null {
  const low = t.toLowerCase();
  if (/\b(much|a lot|a lot more|really|crank|all the way|max(imum)?)\b/.test(low)) return 12;
  if (/\b(a bit|a little|slightly|small|tiny|little)\b/.test(low)) return -3;
  if (/\b(quieter|quieter|softer|too loud|drop it|turn it down|lower)\b/.test(low)) return -6;
  if (/\b(louder|up|bigger|more volume|louden|raise)\b/.test(low)) return 6;
  return null;
}

function detectAbsolute(t: string): number | null {
  const low = t.toLowerCase();
  if (/\bmax(imum)?(\s+volume)?\b/.test(low)) return 12;
  if (/\bmin(imum)?|mute\b/.test(low)) return -30;
  const n = normalizeNumeric(low);
  if (n !== null) {
    // "set volume to 6" -> absolute dB clamp -30..+12; if user means 60%, interpret
    if (/\b(percent|%)\b/.test(low)) {
      if (n <= 100) return Math.round(-30 + (n / 100) * 42);
    }
    return Math.min(12, Math.max(-30, n));
  }
  return null;
}

export async function evaluateVolumeIntent(transcript: string): Promise<VolumeIntent | null> {
  const t = transcript.trim();
  if (!t) return null;
  const now = Date.now();
  if (lastResult && lastTranscript === t && now - lastCallAt < CACHE_WINDOW_MS) return lastResult;

  const prompt = BOOLEAN_PROMPT(t);
  try {
    const { stdout } = await execFileAsync('openclaw', [
      'agent', '--agent', 'main', '--json', '--timeout', '8', '-m', prompt,
    ], { env: process.env });
    // Parse the agent JSON answer; it comes as { text: ... } or a text blob.
    const parsed = JSON.parse(stdout);
    const s = (parsed?.text ?? parsed?.reply ?? '').toLowerCase().trim();
    const yes = /^yes\b/.test(s) || s.includes('"answer":true') || s.includes('"answer": "true"');
    const result: VolumeIntent = {
      requested: yes,
      absoluteDb: detectAbsolute(t),
      adjustDb: yes ? parseRelativeAmount(t) : null,
      probability: yes ? 0.95 : 0.05,   // CLI answer is binary (yes/no); no probability mass
    };
    lastCallAt = now; lastTranscript = t; lastResult = result;
    return result;
  } catch (err) {
    // The detector's failure mode is benign: treat as not-requested and let the
    // chat turn run. Never block voice turns on classification failure.
    const reason = err instanceof Error ? err.message : String(err);
    console.warn(`[volume-intent] evaluation failed (treating as no): ${reason}`);
    lastCallAt = now; lastTranscript = t; lastResult = null;
    return null;
  }
}
