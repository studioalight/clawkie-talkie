/**
 * volumeIntent — decision-model volume-intent detector for voice turns.
 *
 * Design (route 4, clean): the daemon classifies the STT transcript with the
 * local decision model via the typesafe relay loopback
 *   POST http://127.0.0.1:18099/v1/systemone
 * (relay → 192.168.0.149 ollama, model clef-flash). This is a plain HTTP call
 * from the daemon process — NOT an `openclaw agent` spawn — so it avoids the
 * gateway's inter-session attribution/serialization guard that blocked the
 * earlier spawn-based version (the 09:29 `openclaw_gateway_unavailable` failure).
 *
 * On a hit (probabilityTrue >= threshold) the caller applies the gain in-process,
 * gives a short spoken ack, and skips the chat turn entirely. On a miss or any
 * decision-backend failure the normal chat turn proceeds (safe degradation).
 *
 * Rate-limited + cached 6s (volume requests arrive in bursts on repeat PTT).
 *
 * NOTE 2026-10-16: kev-latest on 192.168.0.149 is unloadable by that host's ollama
 * 0.40.5 ("format that is no longer supported", local GGUF import, not re-pullable);
 * Fredrik re-imported clef-flash the same evening and it answers typed decisions
 * correctly (p=0.968 volume / p=0.006 control), so the default model is clef-flash.
 * Until then classify() returns null and the turn falls through.
 */

export interface VolumeIntent {
  requested: boolean;
  absoluteDb: number | null;   // "set volume to N/max/mute"
  adjustDb: number | null;     // "louder"/"quieter"/"a bit"
  probability: number;
}

const ENDPOINT =
  process.env.VOLUME_DECISION_ENDPOINT ?? 'http://127.0.0.1:18099/v1/systemone';
const MODEL = process.env.VOLUME_DECISION_MODEL ?? 'clef-flash';
const THRESHOLD = 0.8;
const TIMEOUT_MS = 25000; // must cover cold model load on .149 (~10-20s)
const CACHE_WINDOW_MS = 6000;
// Keep-warm: ollama unloads idle models after ~5m; probe just under that so
// decision calls stay fast instead of re-paying load on the field.
const WARM_INTERVAL_MS = 240000;
// Circuit breaker: after repeated backend failures stop calling it for a
// cooldown so voice turns never stall on a dead endpoint.
const BREAKER_THRESHOLD = 2;
const BREAKER_COOLDOWN_MS = 600000;

let consecutiveFailures = 0;
let breakerUntil = 0;

let lastCallAt = 0;
let lastTranscript = '';
let lastResult: VolumeIntent | null = null;

const VOLUME_QUESTION =
  'Is the user requesting a change to playback volume (turning it up, turning it down, ' +
  'or setting it to a specific level)? ' +
  'Yes only if the utterance asks for a volume/gain change. ' +
  'No for greetings, questions, or unrelated commands.';

function normalizeNumeric(text: string): number | null {
  const m = text.match(/\b(\d{1,2})\b/i);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  return Number.isFinite(n) ? n : null;
}

function parseRelativeAmount(t: string): number | null {
  const low = t.toLowerCase();
  if (/\b(much|a lot|a lot more|really|crank|all the way|max(imum)?)\b/.test(low)) return 12;
  if (/\b(a bit|a little|slightly|small|tiny|little)\b/.test(low)) return -3;
  if (/\b(quieter|softer|too loud|drop it|turn it down|lower)\b/.test(low)) return -6;
  if (/\b(louder|up|bigger|more volume|louden|raise)\b/.test(low)) return 6;
  return null;
}

function detectAbsolute(t: string): number | null {
  const low = t.toLowerCase();
  if (/\bmax(imum)?(\s+volume)?\b/.test(low)) return 12;
  if (/\bmin(imum)?\b/.test(low) || /\bmute\b/.test(low)) return -30;
  const n = normalizeNumeric(low);
  if (n !== null) {
    if (/\b(percent|%)\b/.test(low)) {
      if (n <= 100) return Math.round(-30 + (n / 100) * 42);
    }
    return Math.min(12, Math.max(-30, n));
  }
  return null;
}

/** One decision-model call. Returns probabilityTrue, or null on any failure. */
async function classify(transcript: string): Promise<number | null> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const body = {
      model: MODEL,
      state: {
        transcript: transcript.slice(0, 300),
        context: 'User speaking to a voice assistant on a clawkie talkie phone.',
      },
      questions: { vol: { type: 'noul', instructions: VOLUME_QUESTION } },
    };
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      console.warn(`[volume-intent] decision endpoint ${res.status}: ${text.slice(0, 200)}`);
      return null;
    }
    const json = await res.json();
    const p = json?.answers?.vol?.noul;
    if (typeof p === 'number') return p;
    console.warn(
      `[volume-intent] unexpected decision response: ${JSON.stringify(json).slice(0, 200)}`,
    );
    return null;
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    console.warn(`[volume-intent] decision call failed (treating as no): ${reason}`);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Tiny no-op evaluation that keeps the decision model loaded on .149. */
async function warmProbe(): Promise<void> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        state: 'warm',
        questions: { warm: { type: 'noul', instructions: 'Warmup probe. Answer 0.' } },
      }),
      signal: ctrl.signal,
    });
    if (res.ok) {
      if (consecutiveFailures > 0) {
        console.log('[volume-intent] backend healthy again (warm probe ok)');
      }
      consecutiveFailures = 0;
    } else {
      console.warn(`[volume-intent] warm probe ${res.status}`);
    }
  } catch {
    console.warn('[volume-intent] warm probe failed (backend unreachable or slow)');
  } finally {
    clearTimeout(timer);
  }
}

// Fire one warm probe at startup, then keep the model resident. unref'd so it
// never holds the process open (daemon runs under launchd anyway).
let warmTimer: ReturnType<typeof setInterval> | null = null;
function ensureWarmLoop(): void {
  if (warmTimer) return;
  warmTimer = setInterval(() => void warmProbe(), WARM_INTERVAL_MS);
  warmTimer.unref?.();
  void warmProbe();
}

export async function evaluateVolumeIntent(transcript: string): Promise<VolumeIntent | null> {
  ensureWarmLoop();
  const t = transcript.trim();
  if (!t) return null;
  if (Date.now() < breakerUntil) return null; // breaker open: chat turn, no stall
  const now = Date.now();
  if (lastResult && lastTranscript === t && now - lastCallAt < CACHE_WINDOW_MS) {
    return lastResult;
  }

  const probability = await classify(t);
  if (probability === null) {
    consecutiveFailures += 1;
    if (consecutiveFailures >= BREAKER_THRESHOLD) {
      breakerUntil = Date.now() + BREAKER_COOLDOWN_MS;
      console.warn(
        `[volume-intent] backend failed ${consecutiveFailures}x — breaker open for ${BREAKER_COOLDOWN_MS / 60000} min (chat turns unaffected)`,
      );
    }
    lastCallAt = now;
    lastTranscript = t;
    lastResult = null;
    return null;
  }
  consecutiveFailures = 0;

  const requested = probability >= THRESHOLD;
  const result: VolumeIntent = {
    requested,
    absoluteDb: requested ? detectAbsolute(t) : null,
    adjustDb: requested ? parseRelativeAmount(t) : null,
    probability,
  };
  lastCallAt = now;
  lastTranscript = t;
  lastResult = result;
  return result;
}
