// VoiceSession state + runtime.
//
// Pure state core (`createVoiceSessionState`) is the part Vitest covers
// — it captures the room/session binding for one active voice
// session and tracks the in-flight-turn / closed flags.
//
// The runtime class (`VoiceSession`) owns the live WebRTC peer, STT,
// TTS, OpenClaw chat, audio plumbing, and keepalive for one
// deterministic per-session room. Multiple `VoiceSession` instances
// coexist inside one daemon process, one per rendezvous-derived
// `roomId`. None of the existing per-turn singleton fields from
// `peer.ts` survive — they all moved here so each room is isolated.

import { wrtc } from './wrtc-patch.js';
import SimplePeer from 'simple-peer';
import { runChat, ChatError, shouldDeliverReplyForChatTarget, type DeliveryTarget as ChatDeliveryTarget } from './chatSession.js';
import { OpenClawInferTtsSession, TTS_SAMPLE_RATE, type TtsSessionCallbacks, type TtsSessionOptions } from './ttsSession.js';
import { ElevenLabsTtsSession } from './elevenLabsTtsSession.js';
import { daemonHandshakeResponse, daemonToPhone, type ControlEventRecord, type DaemonToPhone, type DaemonToPhoneEvent, type NewSessionDestinationOption, type NewSessionDestinationsCatalog, type PhoneToDaemon, type RecentSessionsSnapshot, type SttCatalog, type SttSelection, type TtsCatalog, type TtsSelection, type VoiceSettings, type VoiceTurnSnapshot } from './protocol.js';
import { OpenClawInferSttSession, type OpenClawInferSttSessionOptions } from './inferSttSession.js';
import type { SttSessionCallbacks } from './sttTypes.js';
import { createWasmVad, type SpeechDetector, type WasmVadOptions } from './vad.js';
import { SignalClient, type SignalData } from './signal.js';
import { classifySignal, decideForwardToLivePeer, decideIncomingSignal } from './signalKind.js';
import { createEmptyTtsCatalog, defaultTtsCatalogCache } from './ttsCatalog.js';
import { PushAudioChannel, type PushAudioRequest, type PushAudioResult, type PushSpeechRequest, type PushStreamRequest } from './pushAudio.js';
import type { VoiceSessionNotificationPort } from './voiceSessionPort.js';
import { createEmptySttCatalog, defaultSttCatalogCache } from './sttCatalog.js';
import { createEmptyRecentSessionsSnapshot, defaultRecentSessionsCache } from './recentSessions.js';
import {
  buildNewSessionCreateResponse,
  createWebchatOnlyNewSessionDestinationsCatalog,
  getNewSessionDestinationsWithOpenClaw,
  type NewSessionCreateRequestLike,
} from './newSession.js';

import {
  FRAME_10MS,
  WEBRTC_SAMPLE_RATE,
  pcmToFrames,
  resamplePcm,
} from './audio.js';

const MAX_BUFFERED_CANDIDATES_PER_PEER = 32;
const RECENT_SESSIONS_SUBSCRIPTION_INTERVAL_MS = 60_000;
const MAX_CONTROL_HISTORY_EVENTS = 128;
const MAX_BUFFERED_TTS_TURN_BYTES = 64 * 1024 * 1024;

// Deep-rest boards can hold a phantom turnInFlight (no TTS stream, no agent
// turn) — a queued session switch then delivers after this grace instead of
// waiting forever (2026-10-14, AMOLED deep-rest field case).
const PHANTOM_SWITCH_GRACE_MS = 12_000;
// Realtime pacing for data-channel TTS (2026-10-07 23:3x, rewritten 10-08): the ESP32 speaker
// ring is only 2s (64KB @16kHz) and audio_pipeline_play runs in the esp_peer
// recv callback — blasting ElevenLabs-produced chunks overruns it, blocks the
// peer loop, SCTP flow-control backs up silently, and playback stalls in
// 2s-burst / 30-40s-gap cycles (23:35) — or, with the first counter-based
// attempt, deadlocks entirely (silence, no tts.done: 10-08 09:16 serial).
// Model: a byte budget grows at realtime (TTS_PACE_BYTES_PER_TICK per
// TTS_PACE_TICK_MS, capped at ~1s) and chunks only leave when the budget can
// cover them — generation speed never reaches the wire faster than playback.
// REWRITE 11:19 after field diag (11:16 turn): pump held EXACTLY 1.0x
// realtime with ZERO slack — 145 chunks in 15.1s (0.96x), no stalls, no
// gaps, no failures... and the board still paused. A receiver with no
// buffer slack underruns on the first scheduling jitter and its player
// stall-seeks. Fix: head-start + slight overdrive — board gets ~1.5s of
// buffer immediately, then audio arrives at TTS_PACE_RATE (>1.0x realtime)
// so its buffer stays ~1.5s deep: never starves, never overflows the 2s
// ring (drop-incoming guards the worst case).
const TTS_PACE_TICK_MS = 40;
const TTS_PACE_BYTES_PER_TICK = 1280; // 40ms of 16kHz mono PCM16 per tick (fits Ethernet MTU)
const TTS_PACE_LOOKAHEAD_BYTES = 96000; // budget cap ~3s (field 09:25 option 3: v3 generates ~0.5x realtime; deeper lookahead covers generation stalls so the board's 32KB resume cushion stays in range)
const TTS_PACE_HEADSTART_BYTES = 38400; // cushion: 1.2s headstart matching ESP32 preroll buffer
const TTS_PACE_RATE = 1.1; // 1.1x steady feed to sustain 1.2s buffer depth
const MAX_SCTP_BUFFERED_AMOUNT = 64 * 1024; // 64KB cushion: prevents delayed-SACK micro-pauses while guarding against runaway

export interface VoiceSessionConfig {
  roomId: string;
  sessionId: string;
  sessionKey?: string;
  channel?: string;
  target?: string;
  accountId?: string;
  delivery?: ChatDeliveryTarget;
}

export interface VoiceSessionState {
  roomId: string;
  handleStartTurn(): void;
  resetTurn(): void;
  close(): void;
  readonly turnInFlight: boolean;
  readonly closed: boolean;
  chatTarget(): { sessionId: string; sessionKey?: string; channel?: string; target?: string; accountId?: string; delivery?: ChatDeliveryTarget };
}

export function createVoiceSessionState(config: VoiceSessionConfig): VoiceSessionState {
  let turnInFlight = false;
  let closed = false;

  return {
    roomId: config.roomId,
    handleStartTurn() {
      turnInFlight = true;
    },
    resetTurn() {
      turnInFlight = false;
    },
    close() {
      closed = true;
    },
    get turnInFlight() {
      return turnInFlight;
    },
    get closed() {
      return closed;
    },
    chatTarget() {
      return {
        sessionId: config.sessionId,
        ...(config.sessionKey ? { sessionKey: config.sessionKey } : {}),
        ...(config.channel ? { channel: config.channel } : {}),
        ...(config.target ? { target: config.target } : {}),
        ...(config.accountId ? { accountId: config.accountId } : {}),
        delivery: config.delivery,
      };
    },
  };
}

export type PhoneConnectionDecision = 'accept' | 'use_existing' | 'replace_existing';

export function decidePhoneConnection(input: {
  hasCurrentPeer: boolean;
  currentRemoteId: string | null;
  incomingRemoteId: string;
}): PhoneConnectionDecision {
  if (!input.hasCurrentPeer) return 'accept';
  if (input.currentRemoteId === input.incomingRemoteId) return 'use_existing';
  return 'replace_existing';
}

// --- runtime ---------------------------------------------------------

// Mobile connections can need a longer second-stage WebRTC handshake.
const CONNECT_TIMEOUT_MS = 30_000;
const REPLACED_REMOTE_IGNORE_TTL_MS = 30_000;
const STT_SAMPLE_RATE = 16000;
const TTS_CATALOG_LOAD_TIMEOUT_MS = 1_500;
const OPENAI_PROVIDER_ID = 'openai';
const LEGACY_CLAWKIE_TTS_VOICE_IDS = new Set(['eve', 'ara', 'rex', 'sal', 'leo']);

type SignalPayload = Parameters<SimplePeer.Instance['signal']>[0];

interface AudioSourceLike {
  createTrack(): unknown;
  onData(frame: { samples: Int16Array; sampleRate: number; channelCount: number }): void;
}

interface MediaStreamLike {
  addTrack(track: unknown): void;
}

// Canonical per-turn TTS PCM log. TTS callbacks append here exactly once;
// live data-channel sends, WebRTC frame staging, and reconnect replay all
// consume from this log instead of deciding whether audio should be retained.
interface TtsAudioTurn {
  turnId: number;
  text: string;
  sampleRate: number;
  chunks: Buffer[];
  byteLength: number;
  started: boolean;
  abandoned: boolean;
  complete: boolean;
  drained: boolean;
  overflowed: boolean;
  replayOnReconnect: boolean;
  replayFromChunkIndex: number;
  liveDataCursor: number;
}

export interface SttSessionLike {
  sendAudio(bytes: Uint8Array): void;
  signalAudioDone(): void | Promise<void>;
  close(): void;
}

export type SttSessionFactory = (
  opts: OpenClawInferSttSessionOptions,
  cb: SttSessionCallbacks,
) => SttSessionLike;

export type SpeechDetectorFactory = (opts: WasmVadOptions) => Promise<SpeechDetector>;

export interface TtsSessionLike {
  cancel(): void;
}

export type TtsSessionFactory = (
  opts: TtsSessionOptions,
  cb: TtsSessionCallbacks,
) => TtsSessionLike;

function sanitizeReplyFailureLogText(text: string): string {
  return text
    .replace(/("--message"\s+)"(?:\\.|[^"\\])*"/g, '$1"[redacted]"')
    .replace(/("-m"\s+)"(?:\\.|[^"\\])*"/g, '$1"[redacted]"')
    .replace(/(--message\s+)(?:"(?:\\.|[^"\\])*"|'[^']*'|\S+)/g, '$1[redacted]')
    .replace(/(-m\s+)(?:"(?:\\.|[^"\\])*"|'[^']*'|\S+)/g, '$1[redacted]')
    .replace(/((?:authorization|proxy-authorization)\s*:\s*(?:bearer|basic)\s+)\S+/gi, '$1[redacted]')
    .replace(/([A-Za-z0-9_.-]*(?:api[_-]?key|apikey|secret|token|credential|password)[A-Za-z0-9_.-]*\s*[=:]\s*)(?:"(?:\\.|[^"\\])*"|'[^']*'|\S+)/gi, '$1[redacted]')
    .replace(/\b(token-[A-Za-z0-9_.-]+)\b/gi, '[redacted]')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 1000);
}

function sanitizedErrorMessage(err: unknown): string {
  const parts: string[] = [];
  if (err instanceof Error && err.message) parts.push(err.message);
  if (err && typeof err === 'object') {
    const maybe = err as { stderr?: unknown; stdout?: unknown };
    if (typeof maybe.stderr === 'string' && maybe.stderr.trim()) parts.push(maybe.stderr);
    if (typeof maybe.stdout === 'string' && maybe.stdout.trim()) parts.push(maybe.stdout);
  }
  return sanitizeReplyFailureLogText(parts.join(' ') || 'unknown');
}

export interface VoiceSessionRuntimeOptions {
  sttLanguage?: string;
  signalServer: string;
  iceServers: RTCIceServer[];
  hostPeerId: string;
  roomId: string;
  sessionId: string;
  sessionKey?: string;
  channel?: string;
  target?: string;
  accountId?: string;
  delivery?: ChatDeliveryTarget;
  voiceSettings?: VoiceSettings;
  sttSessionFactory?: SttSessionFactory;
  createSpeechDetector?: SpeechDetectorFactory;
  ttsSessionFactory?: TtsSessionFactory;
  ttsCatalogProvider?: () => Promise<TtsCatalog>;
  sttCatalogProvider?: () => Promise<SttCatalog>;
  recentSessionsProvider?: () => Promise<RecentSessionsSnapshot>;
  newSessionDestinationsProvider?: () => Promise<NewSessionDestinationsCatalog>;
  newSessionDiscordDestinationsProvider?: () => Promise<NewSessionDestinationOption[]>;
  newSessionSlackDestinationsProvider?: () => Promise<NewSessionDestinationOption[]>;
  newSessionCreateResponder?: (msg: NewSessionCreateRequestLike) => Promise<DaemonToPhone>;
  onClose: (roomId: string) => void;
}

export class VoiceSession {
  readonly roomId: string;
  /** OpenClaw session this room speaks for (join target). */
  readonly sessionId: string;
  private readonly state: VoiceSessionState;
  private readonly signalClient: SignalClient;
  private peer: SimplePeer.Instance | null = null;
  private remoteId: string | null = null;
  private peerInitiator = false;
  private acceptedOffer = false;
  private acceptedAnswer = false;
  private connected = false;
  private connectionTimeout: NodeJS.Timeout | null = null;
  private stt: SttSessionLike | null = null;
  private tts: TtsSessionLike | null = null;
  private chatAbort: AbortController | null = null;
  private audioSource: AudioSourceLike | null = null;
  private outboundStream: unknown = null;
  private keepaliveInterval: NodeJS.Timeout | null = null;
  // Resampled WebRTC frame staging for RTCAudioSource; canonical PCM lives in ttsAudioTurn.
  private audioFrameQueue: Int16Array[] = [];
  private audioPumpInterval: NodeJS.Timeout | null = null;
  private audioPumpAwaitingDone = false;
  private rawRemainder: Buffer = Buffer.alloc(0);
  private resampledRemainder: Buffer = Buffer.alloc(0);
  private sampleRateForTts: number = TTS_SAMPLE_RATE; // tracks the active TTS output rate
  private closing = false;
  private ttsSelection: TtsSelection = {};
  private sttSelection: SttSelection = {};
  private sttOpenToken = 0;
  private _pcmDumpPath?: string;
  private _pcmDumpBytes = 0;
  private turnToken = 0;
  private audioPumpTurnToken: number | null = null;
  private ttsPaceTimer: NodeJS.Timeout | null = null;
  private ttsPaceToken: number | null = null;
  private ttsPaceCursor = 0; // next canonical chunk index to send live
  private ttsPaceBudget = 0; // send bytes currently allowed (grows at realtime)
  private ttsPaceLastGrowAt = 0; // wall-clock of last budget growth (time-based pacing, 2026-10-15)
  // TTS field-timing diagnostics (2026-10-08 11:0x): lab tests exonerated the
  // API (4x realtime), resampler (3.8x), board (767e4cf) — but field turns
  // still stall. Instrument the live pump→send path to name the layer.
  private ttsDiagStartedAt: number | null = null;
  private ttsDiagSentCount = 0;
  private ttsDiagLastSendAt = 0;
  private ttsDiagLastTickAt = 0;
  private ttsDiagStarvedLogged = false;
  private ttsPaceDonePending = false; // generation done; tts waits for pump to drain
  private recentSessionsInterval: NodeJS.Timeout | null = null;
  private readonly replacedRemoteIds = new Set<string>();
  private readonly pendingCandidates = new Map<string, SignalPayload[]>();
  private controlEventId = 0;
  private lastPeerDetachEventId = 0;
  private lastPeerDetachAtMs: number | null = null;
  private lastUsedAtMsValue = Date.now();
  private readonly controlHistory: ControlEventRecord[] = [];
  private protocolUnsupported = false;
  private ttsAudioTurn: TtsAudioTurn | null = null;
  private readonly pushChannel = new PushAudioChannel();

  /** Client-drain watermark: epoch ms until the client has plausibly finished
   * PLAYING everything sent so far (bytes consumed at 16k realtime). Updated
   * by noteAudioSent on every binary audio send; cleared by mic-priority and
   * on teardown. The push queue gates on this so notifications never dequeue
   * over a reply's still-draining burst (2026-09-18). */
  private audioDrainUntilMs = 0;

  /** Cumulative audio bytes sent since the current drain anchor. */
  private audioSentSinceAnchorBytes = 0;
  /** When the current drain backlog began (first byte after idle). */
  private audioDrainAnchorMs = 0;

  /** Track sent-audio bytes for the drain watermark. Chunks arrive in bursts
   * faster than realtime (26s of speech sent in ~6s), so the watermark must
   * measure the CUMULATIVE backlog since an anchor — a per-chunk duration
   * never adds up and the queue releases ~19s early (field 2026-09-18:
   * radio started at 10:34:57 while the speech played until 10:35:16). */
  private noteAudioSent(byteLength: number): void {
    if (byteLength <= 0) return;
    const now = Date.now();
    if (this.audioDrainUntilMs <= now) {
      // Previous backlog fully drained (or none) — re-anchor at now.
      this.audioSentSinceAnchorBytes = 0;
      this.audioDrainAnchorMs = now;
    }
    this.audioSentSinceAnchorBytes += byteLength;
    const playMs = (this.audioSentSinceAnchorBytes / (STT_SAMPLE_RATE * 2)) * 1000;
    this.audioDrainUntilMs = this.audioDrainAnchorMs + playMs;
  }
  private turnSnapshot: VoiceTurnSnapshot = {
    inFlight: false,
    phase: 'idle',
    latestEventId: 0,
  };

  constructor(private readonly opts: VoiceSessionRuntimeOptions) {
    this.roomId = opts.roomId;
    this.sessionId = opts.sessionId;
    this.ttsSelection = normalizeTtsSelection(opts.voiceSettings);
    this.sttSelection = normalizeSttSelection(opts.voiceSettings);
    this.state = createVoiceSessionState({
      roomId: opts.roomId,
      sessionId: opts.sessionId,
      ...(opts.sessionKey ? { sessionKey: opts.sessionKey } : {}),
      ...(opts.channel ? { channel: opts.channel } : {}),
      ...(opts.target ? { target: opts.target } : {}),
      ...(opts.accountId ? { accountId: opts.accountId } : {}),
      delivery: opts.delivery,
    });

    // Signaling identity for the daemon side of this voice room. Each
    // VoiceSession joins its own deterministic per-session room so
    // browsers landing on `host:session-A` and `host:session-B` reach
    // independent peer/STT/TTS/chat lanes.
    this.signalClient = new SignalClient({
      signalServer: opts.signalServer,
      peerId: opts.roomId,
      roomName: opts.roomId,
    });

    this.signalClient.on('error', (err) => {
      console.error(`[voice ${opts.roomId}] signal error: ${err.message}`);
    });

    this.signalClient.on('announce', ({ peerId }) => {
      console.log(`[voice ${opts.roomId}] DEBUG: announce from ${peerId} (replaced=${this.replacedRemoteIds.has(peerId)})`);
      if (this.replacedRemoteIds.has(peerId)) return;
      this.acceptPhone(peerId, true);
    });

    this.signalClient.on('signal', async (event) => {
      console.log(`[voice ${opts.roomId}] DEBUG: received signal from ${event.from} (remoteId=${this.remoteId}, hasPeer=${!!this.peer}, peerDestroyed=${this.peer?.destroyed})`);
      if (this.replacedRemoteIds.has(event.from)) return;
      const payload = event.data as SignalPayload;

      /* Check if this is a data-channel-over-signaling message (stt.*, etc.) */
      if (payload && typeof payload === 'object' && 't' in payload && typeof (payload as Record<string, unknown>).t === 'string') {
        const t = (payload as Record<string, unknown>).t as string;
        if (t === 'stt.start' || t === 'stt.audio' || t === 'stt.audio.done' || t === 'stt.cancel' || t === 'client.hello' || t === 'reply.cancel' || t === 'settings.update') {
          console.log(`[voice ${opts.roomId}] DEBUG: data-channel-over-signal t=${t}`);
          if (t === 'stt.audio') {
            /* Decode base64 audio and feed to STT */
            const b64 = (payload as Record<string, unknown>).b64 as string | undefined;
            if (b64 && this.stt) {
              const audioBytes = Buffer.from(b64, 'base64');
              this.stt.sendAudio(new Uint8Array(audioBytes));
              /* Debug: append PCM to file for validation */
              const fs = await import('node:fs');
              if (!this._pcmDumpPath) {
                this._pcmDumpPath = `/tmp/clawkie-debug-${Date.now()}.pcm`;
                this._pcmDumpBytes = 0;
              }
              fs.appendFileSync(this._pcmDumpPath, audioBytes);
              this._pcmDumpBytes += audioBytes.length;
              console.log(`[voice ${opts.roomId}] DEBUG: stt.audio chunk=${audioBytes.length} total=${this._pcmDumpBytes} file=${this._pcmDumpPath}`);
            } else if (!b64) {
              console.log(`[voice ${opts.roomId}] WARNING: stt.audio missing b64 field`);
            } else if (!this.stt) {
              console.log(`[voice ${opts.roomId}] WARNING: stt.audio received but no STT session`);
            }
          } else {
            /* Feed control message to handleControl */
            this.handleControl(payload as unknown as PhoneToDaemon);
          }
          return;
        }
      }

      const livePeer = !!this.peer && this.remoteId === event.from && !this.peer.destroyed;
      console.log(`[voice ${opts.roomId}] DEBUG incoming signal keys: ${Object.keys(payload ?? {})}, type=${(payload as any)?.type}, cand=${!!(payload as any)?.candidate}`);
      const kind = classifySignal(payload);
      console.log(`[voice ${opts.roomId}] DEBUG: signal kind=${kind} from ${event.from} livePeer=${livePeer}`);
      const action = decideIncomingSignal({ hasLivePeer: livePeer, kind });
      console.log(`[voice ${opts.roomId}] DEBUG: action=${action}`);

      if (action === 'forward') {
        const decision = decideForwardToLivePeer(
          {
            initiator: this.peerInitiator,
            acceptedOffer: this.acceptedOffer,
            acceptedAnswer: this.acceptedAnswer,
          },
          kind,
        );
        if (decision !== 'forward') {
          console.error(`[voice ${opts.roomId}] dropping ${kind} from ${event.from}: ${decision}`);
          return;
        }
        try {
          this.peer!.signal(payload);
          if (kind === 'offer') this.acceptedOffer = true;
          if (kind === 'answer') this.acceptedAnswer = true;
        } catch (err) {
          console.error(`[voice ${opts.roomId}] peer.signal failed: ${err instanceof Error ? err.message : err}`);
        }
        return;
      }

      if (action === 'create-non-initiator') {
        const buffered = this.pendingCandidates.get(event.from) ?? [];
        this.pendingCandidates.delete(event.from);
        this.acceptPhone(event.from, false, payload);
        if (this.peer && this.remoteId === event.from) {
          for (const cand of buffered) {
            try { this.peer.signal(cand); } catch (err) {
              console.error(`[voice ${opts.roomId}] replay candidate failed: ${err instanceof Error ? err.message : err}`);
            }
          }
        }
        return;
      }

      if (action === 'buffer-candidate') {
        const list = this.pendingCandidates.get(event.from) ?? [];
        if (list.length >= MAX_BUFFERED_CANDIDATES_PER_PEER) return;
        list.push(payload);
        this.pendingCandidates.set(event.from, list);
        return;
      }

      console.error(`[voice ${opts.roomId}] ignoring ${kind} signal from ${event.from} with no live peer`);
    });

    this.signalClient.subscribe();
  }

  applyVoiceSettings(settings: VoiceSettings | null | undefined, force = false): void {
    this.touchActivity();
    if (force) {
      // Explicit reset from the control port: replace the selection entirely.
      this.ttsSelection = normalizeTtsSelection(settings);
      this.sttSelection = normalizeSttSelection(settings);
      return;
    }
    // Merge, don't replace: an incoming settings.update WITHOUT a voice (or
    // model) means "no preference" — the client is re-declaring its baseline
    // on every lane rebuild, not asking for a reset. Port-applied voices
    // survive rebuilds this way (field-verified 2026-09-25: the Pi's
    // voiceless settings.update on data channel open otherwise wiped the
    // dynamically-set voice back to the daemon default mid-conversation).
    // Explicit reset = the control port's reset:true.
    const incoming = normalizeTtsSelection(settings);
    this.ttsSelection = {
      providerId: incoming.providerId ?? this.ttsSelection.providerId,
      model: incoming.model ?? this.ttsSelection.model,
      voice: incoming.voice ?? this.ttsSelection.voice,
    };
    this.sttSelection = normalizeSttSelection(settings);
  }

  // --- Live TTS volume (voice control port) --------------------------------
  // Session state in dB, applied in the daemon's PCM forward path so every
  // client (Pi, web, ESP32) receives pre-scaled audio — device-agnostic by
  // construction. 0 = neutral: field-tuned device calibration on the client
  // side is untouched. ElevenLabs exposes no volume parameter in
  // voice_settings, so the code framework is the right (and only) layer.
  private ttsGainDb = 0;

  // -- switchboard relay (phase one, 2026-09-28) --
  // Operator model: the agent says "switch to D'ENT" → the control port sets
  // a pending line switch DURING the reply; completeTtsTurn relays
  // line.switch to the phone AFTER the reply finished playing, so the
  // operator's confirmation is never cut off by its own command.
  private pendingLineSwitch: string | null = null;
  private pendingSessionSwitch: { session: string; sessionKey?: string } | null = null;
  /** Phantom-turn guard: if the "current turn" never starts streaming TTS
   *  nor runs the agent (deep-rest board with a stuck turnInFlight), the
   *  queued switch would wait forever. One-shot timer per queue. */
  private pendingSwitchGuard: NodeJS.Timeout | null = null;

  /** Session switch (operator model, mirror of line switch): re-aim the
   *  phone at a different session ON THIS SHORE without a restart — the
   *  fresh-session test (2026-09-30) proved the phone treats sessions as
   *  swappable rooms; this verb makes the move spoken and seamless. The
   *  optional sessionKey rides the message so no gateway lookup is needed
   *  on the next join (same fix as the five-layer nest). */
  requestSessionSwitch(session: string, sessionKey?: string): void {
    this.touchActivity();
    const id = session.trim();
    if (!id) return;
    // KEY-DRIFT GUARD (field 22:42, "it created a new session again"): an
    // explicit:<sessionId> key has no external reply channel — the gateway
    // derives it from bare CLI agent turns and it poisons the board (every
    // turn fails openclaw_delivery_unresolved, the room "drifts" and looks
    // like a new session). Reject the key, keep the room's own binding;
    // the switch still proceeds, delivery falls back to TTS-only.
    if (sessionKey?.trim().startsWith('agent:main:explicit:')) {
      console.warn(`[voice ${this.roomId}] session.switch carried an explicit-channel key — rejecting it (key-drift guard), keeping room key`);
      sessionKey = undefined;
    }
    const sw = { session: id, ...(sessionKey?.trim() ? { sessionKey: sessionKey.trim() } : {}) };
    if (!this.state.turnInFlight && this.connected && this.peer && !this.peer.destroyed) {
      console.log(`[voice ${this.roomId}] Session switch → ${id} (idle — delivering now)`);
      this.sendToPeer(this.peer, daemonToPhone.sessionSwitch(id, sessionKey?.trim() || undefined));
      return;
    }
    this.pendingSessionSwitch = sw;
    console.log(`[voice ${this.roomId}] Session switch queued → ${id} (after current turn)`);
    if (!this.pendingSwitchGuard) {
      this.pendingSwitchGuard = setTimeout(() => this.deliverPendingSwitchIfPhantom(), PHANTOM_SWITCH_GRACE_MS);
      this.pendingSwitchGuard.unref?.();
    }
  }

  /** Device-lane salvage (2026-10-14): hand off a pending switch before the
   *  room is closed by lane hygiene — the queue otherwise dies with the room
   *  and the board never rehomes. Idempotent. */
  takePendingSessionSwitch(): { session: string; sessionKey?: string } | null {
    const sw = this.pendingSessionSwitch;
    this.pendingSessionSwitch = null;
    if (this.pendingSwitchGuard) {
      clearTimeout(this.pendingSwitchGuard);
      this.pendingSwitchGuard = null;
    }
    return sw;
  }

  /** Phantom-turn guard (2026-10-14, AMOLED deep rest): the 20:52 switch
   *  queued behind a turnInFlight that had no TTS session and no agent turn
   *  running — a turn that could never complete, so the switch waited
   *  forever. If the queued switch is still pending after the grace and
   *  neither a TTS stream nor an agent turn is alive, the turn is phantom:
   *  reset the state machine and deliver immediately. Idempotent — no-ops
   *  when the switch was already delivered or a real turn is active. */
  private deliverPendingSwitchIfPhantom(): void {
    this.pendingSwitchGuard = null;
    if (!this.pendingSessionSwitch) {
      console.log(`[voice ${this.roomId}] phantom guard: settled — switch already delivered`);
      return;
    }
    if (this.tts || this.chatAbort) {
      console.log(`[voice ${this.roomId}] phantom guard: real turn alive (tts=${!!this.tts} chat=${!!this.chatAbort}) — retry armed`);
      this.pendingSwitchGuard = setTimeout(() => this.deliverPendingSwitchIfPhantom(), PHANTOM_SWITCH_GRACE_MS);
      this.pendingSwitchGuard.unref?.();
      return;
    }
    if (!this.connected || !this.peer || this.peer.destroyed) {
      console.log(`[voice ${this.roomId}] phantom guard: peer gone (connected=${this.connected}) — dropping guard, delivery on next join`);
      return;
    }
    const sw = this.pendingSessionSwitch;
    this.pendingSessionSwitch = null;
    this.state.resetTurn();                               // clear the phantom turnInFlight
    console.log(`[voice ${this.roomId}] Delivering queued session switch (phantom-turn guard) → ${sw.session}`);
    this.sendToPeer(this.peer, daemonToPhone.sessionSwitch(sw.session, sw.sessionKey));
  }

  /** Queue a line switch for delivery when the current TTS turn completes.
   *  Called by the voice control port (agent operator). Raw line name
  *  pass-through — the phone's address book owns the lookup. */
  requestLineSwitch(line: string): void {
    this.touchActivity();
    const name = line.trim();
    if (!name) return;
    // If no TTS turn is in flight, deliver immediately — the operator may
    // queue the switch while idle (e.g. answering a written question about
    // the procedure), and completeTtsTurn never fires, so the switch would
    // sit in memory forever (field-observed on VM3 2026-09-28: D'ENT queued
    // while idle, phone never re-homed).
    if (!this.state.turnInFlight && this.connected && this.peer && !this.peer.destroyed) {
      console.log(`[voice ${this.roomId}] Line switch → ${name} (idle — delivering now)`);
      this.sendToPeer(this.peer, daemonToPhone.lineSwitch(name));
      return;
    }
    this.pendingLineSwitch = name;
    console.log(`[voice ${this.roomId}] Line switch queued → ${name} (after current turn)`);
  }

  /** Set absolute TTS gain (dB), clamped to [-30, +12]. Returns the value
   *  actually in effect. */
  applyTtsVolume(gainDb: number): number {
    const clamped = Math.min(12, Math.max(-30, Number.isFinite(gainDb) ? gainDb : 0));
    if (clamped !== this.ttsGainDb) {
      this.ttsGainDb = clamped;
      console.log('[voice ' + this.roomId + '] TTS volume -> ' + (clamped >= 0 ? '+' : '') + clamped.toFixed(1) + ' dB');
    }
    return clamped;
  }

  /** Relative TTS gain step (dB). Returns the clamped value in effect. */
  adjustTtsVolume(deltaDb: number): number {
    return this.applyTtsVolume(this.ttsGainDb + deltaDb);
  }

  get ttsVolumeDb(): number {
    return this.ttsGainDb;
  }

  /** Scale S16_LE PCM by the session TTS gain; untouched at 0 dB. */
  private gainPcm(pcm: Uint8Array): Buffer {
    if (this.ttsGainDb === 0 || pcm.length < 2) {
      return Buffer.isBuffer(pcm) ? pcm : Buffer.from(pcm);
    }
    const src = Buffer.isBuffer(pcm) ? pcm : Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    const k = Math.pow(10, this.ttsGainDb / 20);
    const out = Buffer.allocUnsafe(src.length);
    for (let i = 0; i + 1 < src.length; i += 2) {
      const s = Math.round(src.readInt16LE(i) * k);
      out.writeInt16LE(s > 32767 ? 32767 : s < -32768 ? -32768 : s, i);
    }
    return out;
  }

  touchActivity(atMs = Date.now()): void {
    this.lastUsedAtMsValue = atMs;
  }

  get lastUsedAtMs(): number {
    return this.lastUsedAtMsValue;
  }

  get canEvictForVoiceSessionLimit(): boolean {
    const hasLivePeer = !!this.peer && !this.peer.destroyed;
    return !this.closing && !hasLivePeer && !this.connected && !this.state.turnInFlight;
  }

  // Test/manager hooks so callers can inspect the selection currently
  // applied to the next TTS turn.
  get currentTtsSelection(): TtsSelection {
    return { ...this.ttsSelection };
  }

  get currentTtsVoice(): string | undefined {
    return this.ttsSelection.voice;
  }

  close(): void {
    if (this.closing) return;
    this.closing = true;
    this.pushChannel.close();
    this.resetTurn('voice_session_closed');
    try {
      this.peer?.destroy();
    } catch {
      // ignore
    }
    this.peer = null;
    this.remoteId = null;
    this.connected = false;
    this.pendingCandidates.clear();
    this.clearConnectionTimeout();
    this.stopKeepalive();
    this.stopRecentSessionsSubscription();
    this.closeOutboundAudio();
    try {
      this.signalClient.close();
    } catch {
      // ignore
    }
    this.state.close();
    this.opts.onClose(this.roomId);
  }

  /**
   * Push unsolicited audio (e.g. a local workspace file) to the connected
   * client, outside any chat turn. Queues while the room is busy so it
   * never interrupts a turn or an open mic. See pushAudio.ts for rules.
   */
  pushAudioFile(req: PushAudioRequest): Promise<PushAudioResult> {
    return this.pushChannel.pushFile(this.notificationPort(), req);
  }

  /** Speak unsolicited text over the notification wire (streaming ElevenLabs). */
  pushSpeech(req: PushSpeechRequest): Promise<PushAudioResult> {
    return this.pushChannel.pushSpeech(this.notificationPort(), req);
  }

  /** Stream a live audio URL as a notification (30-min cap). */
  pushStream(req: PushStreamRequest): Promise<PushAudioResult> {
    return this.pushChannel.pushStream(this.notificationPort(), req);
  }

  /** Stop the in-flight notification push (e.g. client triple-tap stop). */
  stopPush(): boolean {
    return this.pushChannel.requestStop();
  }

  private notificationPort(): VoiceSessionNotificationPort {
    return {
      isRoomQuiet: () =>
        !this.closing && !this.state.turnInFlight && this.stt === null &&
        // Client drain awareness: burst-delivered audio (a reply sent in ~2s)
        // keeps PLAYING on the client long after the daemon finished sending.
        // A notification dequeued the moment sending ends plays over the tail
        // (field 2026-09-18: radio started while the reply was still sounding
        // — 'everything playing at the same time'). Hold the queue until the
        // client has consumed the audio at realtime (see noteAudioSent).
        Date.now() >= this.audioDrainUntilMs,
      isConnected: () =>
        !this.closing && !this.protocolUnsupported && this.connected && !!this.peer && !this.peer.destroyed,
      // Connection-scoped (not recorded into catch-up history): a client
      // that reconnects missed the notification — that is correct;
      // notifications are ephemeral by design.
      sendControl: (msg) => {
        if (msg.t === 'session.snapshot') return false;
        const peer = this.peer;
        if (!peer || !this.connected) return false;
        return this.sendToPeer(peer, msg);
      },
      sendAudio: (pcm) => this.sendBinary(this.gainPcm(pcm)),
    };
  }

  private acceptPhone(remoteId: string, initiator: boolean, initialSignal?: SignalPayload): void {
    if (this.replacedRemoteIds.has(remoteId)) return;
    this.touchActivity();

    const decision = decidePhoneConnection({
      hasCurrentPeer: !!this.peer && !this.peer.destroyed,
      currentRemoteId: this.remoteId,
      incomingRemoteId: remoteId,
    });

    if (decision === 'use_existing') {
      if (initialSignal) {
        try { this.peer?.signal(initialSignal); } catch { /* ignore */ }
      }
      return;
    }

    if (decision === 'replace_existing') {
      if (!this.connected) {
        console.error(`[voice ${this.roomId}] replacing stale pending phone ${this.remoteId ?? 'unknown'} with ${remoteId}`);
        this.replaceCurrentPhone('replace_stale_pending_phone');
      } else {
        console.error(`[voice ${this.roomId}] replacing active phone ${this.remoteId ?? 'unknown'} with ${remoteId}`);
        this.replaceCurrentPhone('newer_phone_connected');
      }
    }

    this.protocolUnsupported = false;
    this.remoteId = remoteId;
    this.peerInitiator = initiator;
    this.acceptedOffer = false;
    this.acceptedAnswer = false;
    console.error(`[voice ${this.roomId}] establishing connection with phone=${remoteId} initiator=${initiator}`);

    console.log(`[voice ${this.roomId}] ICE CONFIG: ${JSON.stringify(this.opts.iceServers)}`);
    const { stream } = this.openOutboundAudio();

    const peer = new SimplePeer({
      initiator,
      trickle: true,
      wrtc: wrtc as unknown as SimplePeer.Options['wrtc'],
      // ICE-lite munge (2026-10-07): libpeer's receive path can't answer
      // daemon checks through TURN (coturn rb>0, sb=0). A lite agent NEVER
      // sends checks — it only answers — so declaring a=ice-lite makes
      // libpeer the sole full agent, driving nomination from its (working)
      // send path. LiveKit (pion, full ICE) answered fine; leaner simple-
      // peer did not. This is the daemon-side workaround while Espressif
      // fixes esp_peer (issue #208).
      // NOTE 2026-10-07 evening: the munge was originally expressed as a
      // sdpTransform option — but simple-peer applies that to the offer
      // object and then emits `this._pc.localDescription || offer`; the
      // node-datachannel polyfill regenerates localDescription from its
      // internal state, so the munge NEVER reached the wire (confirmed via
      // ESP32 serial capture of an ice-lite-less offer). Applied at the
      // signal-handler (wire) level below instead.
      config: {
        iceServers: this.opts.iceServers,
        // Relay-only (plan A, 2026-10-06): same env knob as peer.ts — the
        // daemon's mature stack drives pairing through TURN when enabled,
        // routing around the ESP32's libpeer controlled-role relay gap.
        ...(process.env.CT_ICE_RELAY_ONLY === '1' ? { iceTransportPolicy: 'relay' } : {}),
      },
      streams: stream ? [stream as MediaStream] : undefined,
    });
    this.peer = peer;
    this.armConnectionTimeout(peer, remoteId);

    peer.on('signal', (data) => {
      console.log(`[voice ${this.roomId}] OUTBOUND SIGNAL: ${(data as any)?.type ?? ((data as any)?.candidate ? 'candidate ' + (data as any).candidate.slice(0, 50) : '?')}`);
      if (this.peer !== peer || this.remoteId !== remoteId) return;
      // ICE-lite munge DISABLED (2026-10-08 11:5x): on 1.5.6 a=ice-lite makes
      // libpeer's CONTROLLING agent INITIATE (board sent its own offers —
      // daemon dropped them, every session timed out, serial 11:49-11:52).
      // On direct LAN pairing both agents are full and both send paths work
      // (PeerBinding recv → Send binding response in every boot) — the
      // munge only forced harmful role games. It was a workaround for the
      // TURN-relay recv defect (#208); if remote-wifi relay pairing needs
      // it again, re-enable behind an env flag instead.
      // (block removed — offers now go out un-munged)
      void this.signalClient
        .sendSignal(remoteId, data as unknown as SignalData)
        .catch((err) => {
          console.error(`[voice ${this.roomId}] sendSignal failed: ${err instanceof Error ? err.message : err}`);
        });
    });

    peer.on('connect', () => {
      if (this.peer !== peer) return;
      this.touchActivity();
      this.clearConnectionTimeout();
      this.connected = true;
      console.error(`[voice ${this.roomId}] data channel connected`);
      this.startKeepalive();
      this.sendCatchUpSnapshot(peer);
      // Switch queue flush at data-channel open (2026-10-14 retention test):
      // the board's channel takes ~15s to establish — OLDER than the 12s
      // phantom-guard grace, so switches queued during that window bailed
      // "peer gone" before the board ever heard them. The fresh channel is
      // the natural delivery point: turnInFlight is false, board is listening.
      const sw = this.takePendingSessionSwitch();
      if (sw) {
        console.log(`[voice ${this.roomId}] Delivering queued session switch (data-channel open) → ${sw.session}`);
        this.sendToPeer(this.peer, daemonToPhone.sessionSwitch(sw.session, sw.sessionKey));
      }
      if (this.pendingLineSwitch) {
        const line = this.pendingLineSwitch;
        this.pendingLineSwitch = null;
        console.log(`[voice ${this.roomId}] Delivering queued line switch (data-channel open) → ${line}`);
        this.sendToPeer(this.peer, daemonToPhone.lineSwitch(line));
      }
    });

    peer.on('data', (data: unknown) => {
      if (this.peer !== peer) return;
      this.handlePeerData(data);
    });

    peer.on('close', () => {
      if (this.peer !== peer) return;
      console.error(`[voice ${this.roomId}] data channel closed`);
      this.tearDownPeer('peer_closed', peer);
    });

    peer.on('error', (err: Error) => {
      if (this.peer !== peer) return;
      console.error(`[voice ${this.roomId}] error: ${err.message}`);
      this.tearDownPeer('peer_error', peer);
    });

    if (initialSignal) {
      try {
        peer.signal(initialSignal);
        const initialKind = classifySignal(initialSignal);
        if (initialKind === 'offer') this.acceptedOffer = true;
        if (initialKind === 'answer') this.acceptedAnswer = true;
      } catch (err) {
        console.error(`[voice ${this.roomId}] peer.signal (initial) failed: ${err instanceof Error ? err.message : err}`);
      }
    }
  }

  private replaceCurrentPhone(reason: string): void {
    const previousPeer = this.peer;
    const previousRemoteId = this.remoteId;
    if (!previousPeer) return;

    if (previousRemoteId) this.ignoreReplacedRemote(previousRemoteId);
    if (this.connected && !previousPeer.destroyed) {
      this.sendToPeer(previousPeer, daemonToPhone.sessionReplaced(reason));
    }

    this.clearConnectionTimeout();
    this.stopKeepalive();
    this.stopRecentSessionsSubscription();
    this.closeOutboundAudio();
    this.peer = null;
    this.remoteId = null;
    this.connected = false;
    this.protocolUnsupported = false;
    this.lastPeerDetachEventId = this.controlEventId;
    this.lastPeerDetachAtMs = Date.now();

    setTimeout(() => {
      try {
        previousPeer.destroy();
      } catch {
        // ignore
      }
    }, 100).unref?.();
  }

  private ignoreReplacedRemote(remoteId: string): void {
    this.replacedRemoteIds.add(remoteId);
    this.pendingCandidates.delete(remoteId);
    setTimeout(() => {
      this.replacedRemoteIds.delete(remoteId);
    }, REPLACED_REMOTE_IGNORE_TTL_MS).unref?.();
  }

  private armConnectionTimeout(peer: SimplePeer.Instance, remoteId: string): void {
    this.clearConnectionTimeout();
    this.connectionTimeout = setTimeout(() => {
      if (this.peer !== peer || this.remoteId !== remoteId || this.connected || peer.destroyed) return;
      console.error(`[voice ${this.roomId}] connection timed out with phone=${remoteId}`);
      this.tearDownPeer('connect_timeout', peer);
    }, CONNECT_TIMEOUT_MS);
    this.connectionTimeout.unref?.();
  }

  private clearConnectionTimeout(): void {
    if (!this.connectionTimeout) return;
    clearTimeout(this.connectionTimeout);
    this.connectionTimeout = null;
  }

  private tearDownPeer(reason: string, peer?: SimplePeer.Instance): void {
    if (peer && this.peer !== peer) return;
    this.touchActivity();
    console.error(`[voice ${this.roomId}] tearDownPeer reason=${reason}`);
    this.lastPeerDetachEventId = this.controlEventId;
    this.lastPeerDetachAtMs = Date.now();
    this.clearConnectionTimeout();
    this.stopKeepalive();
    this.stopRecentSessionsSubscription();
    try {
      this.peer?.destroy();
    } catch {
      // ignore
    }
    this.closeOutboundAudio();
    this.peer = null;
    this.remoteId = null;
    this.connected = false;
    this.protocolUnsupported = false;
  }

  private openOutboundAudio(): { stream: unknown | null } {
    // TEMPORARILY DISABLED: Audio track causes wrtc to add m=audio to SDP offer,
    // which triggers ICE failure with esp_peer. Audio is sent via data channel instead.
    // node-datachannel polyfill doesn't support addTrack/RTCAudioSource.
    // Audio is sent via data channel (binary PCM) instead of RTP tracks.
    console.log(`[voice ${this.roomId}] outbound audio track disabled (using data channel for audio)`);
    return { stream: null };
  }

  private closeOutboundAudio(): void {
    if (this.state.turnInFlight && !this.closing) {
      const turn = this.ttsAudioTurn;
      if (turn?.turnId === this.turnToken) {
        if (turn.started) {
          this.abandonTtsAudioTurn(turn.turnId);
        } else {
          this.markTtsAudioReplayOnReconnect(turn.turnId, 0);
        }
      }
    }
    const shouldCompletePendingTrackTts = this.audioPumpAwaitingDone && this.state.turnInFlight && !this.closing;
    this.stopAudioPump();
    this.audioPumpTurnToken = null;
    this.audioFrameQueue = [];
    this.rawRemainder = Buffer.alloc(0);
    this.resampledRemainder = Buffer.alloc(0);
    this.audioSource = null;
    this.outboundStream = null;
    if (shouldCompletePendingTrackTts) {
      this.completeTtsTurn('tts_audio_transport_closed');
    }
  }

  private enqueueTtsPcm(pcm: Uint8Array): void {
    if (!this.audioSource) return;
    if (pcm.byteLength === 0) return;
    const chunk = Buffer.concat([this.rawRemainder, Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength)]);
    const evenLen = chunk.length & ~1;
    this.rawRemainder = chunk.subarray(evenLen);
    if (evenLen === 0) return;
    const aligned = chunk.subarray(0, evenLen);

    const resampled = resamplePcm(aligned, TTS_SAMPLE_RATE, WEBRTC_SAMPLE_RATE);
    const merged = Buffer.concat([this.resampledRemainder, resampled]);
    const frameBytes = FRAME_10MS * 2;
    const wholeBytes = merged.length - (merged.length % frameBytes);
    if (wholeBytes > 0) {
      const framed = merged.subarray(0, wholeBytes);
      const frames = pcmToFrames(framed, FRAME_10MS);
      for (const f of frames) this.audioFrameQueue.push(f);
    }
    this.resampledRemainder = merged.subarray(wholeBytes);
    this.startAudioPump(this.turnToken);
  }

  private flushTtsTail(): void {
    if (this.rawRemainder.length >= 2) {
      const evenLen = this.rawRemainder.length & ~1;
      const aligned = this.rawRemainder.subarray(0, evenLen);
      const resampled = resamplePcm(aligned, TTS_SAMPLE_RATE, WEBRTC_SAMPLE_RATE);
      this.resampledRemainder = Buffer.concat([this.resampledRemainder, resampled]);
      this.rawRemainder = Buffer.alloc(0);
    }
    if (this.resampledRemainder.length >= 2) {
      const tailFrames = pcmToFrames(this.resampledRemainder, FRAME_10MS);
      for (const f of tailFrames) this.audioFrameQueue.push(f);
      this.resampledRemainder = Buffer.alloc(0);
    }
  }

  private startAudioPump(token: number): void {
    if (this.audioPumpInterval) return;
    if (!this.audioSource) return;
    this.audioPumpTurnToken = token;
    this.audioPumpInterval = setInterval(() => {
      if (this.audioPumpTurnToken !== token || !this.isTurnActive(token)) {
        this.stopAudioPump();
        return;
      }
      const source = this.audioSource;
      if (!source) {
        this.stopAudioPump();
        return;
      }
      const frame = this.audioFrameQueue.shift();
      if (!frame) {
        if (this.audioPumpAwaitingDone) {
          this.stopAudioPump();
          this.completeTtsTurn('tts_audio_pump_done');
        }
        return;
      }
      try {
        source.onData({ samples: frame, sampleRate: WEBRTC_SAMPLE_RATE, channelCount: 1 });
        this.markTtsAudioStarted(token);
      } catch (err) {
        console.error(`[voice ${this.roomId}] audioSource.onData failed: ${sanitizedErrorMessage(err)}`);
      }
    }, 10);
    this.audioPumpInterval.unref?.();
  }

  private stopAudioPump(): void {
    if (this.audioPumpInterval) {
      clearInterval(this.audioPumpInterval);
      this.audioPumpInterval = null;
      this.audioPumpTurnToken = null;
    }
  }

  private startKeepalive(): void {
    if (this.keepaliveInterval) return;
    this.keepaliveInterval = setInterval(() => {
      // Keepalive: send every 15s when idle; pause during active audio turns to avoid data channel traffic
      if (!this.connected || !this.peer || this.state.turnInFlight) return;
      try {
        this.peer.send(JSON.stringify({ t: 'keepalive' }));
      } catch (err) {
        console.error(`[voice ${this.roomId}] keepalive send failed: ${err instanceof Error ? err.message : err}`);
      }
    }, 15_000);
    this.keepaliveInterval.unref?.();
  }

  private stopKeepalive(): void {
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval);
      this.keepaliveInterval = null;
    }
  }

  private handlePeerData(data: unknown): void {
    this.touchActivity();
    const bytes = toBytes(data);
    if (!bytes) return;
    // Debug: log first few bytes to understand what's arriving
    if (bytes.length < 50 && bytes[0] !== 0x7b) {
      console.log(`[voice ${this.roomId}] DEBUG: small non-JSON frame (${bytes.length} bytes) firstByte=0x${bytes[0]?.toString(16)} data=${Buffer.from(bytes).toString('hex').substring(0, 40)}`);
    }
    const text = tryDecodeJsonText(bytes);
    if (text !== null) {
      let msg: PhoneToDaemon;
      try {
        msg = JSON.parse(text) as PhoneToDaemon;
      } catch {
        console.log(`[voice ${this.roomId}] DEBUG: JSON parse failed (${bytes.length} bytes): ${text.substring(0, 100)}`);
        return;
      }
      console.log(`[voice ${this.roomId}] DEBUG: handleControl t=${msg.t} stt=${!!this.stt}`);
      this.handleControl(msg);
      return;
    }
    if (this.protocolUnsupported) return;
    if (this.stt) {
      this.stt.sendAudio(bytes);
    } else {
      console.log(`[voice ${this.roomId}] DEBUG: binary audio but stt null (${bytes.length} bytes) firstByte=0x${bytes[0]?.toString(16)}`);
    }
  }

  private async handleControl(msg: PhoneToDaemon): Promise<void> {
    if (this.protocolUnsupported) return;
    if (msg.t === 'client.hello') {
      const response = daemonHandshakeResponse(msg);
      this.sendConnectionScoped(response);
      if (response.t === 'daemon.unsupported') this.enterProtocolUnsupported();
      return;
    }
    if (msg.t === 'tts.catalog.request') {
      void this.sendTtsCatalog();
      return;
    }
    if (msg.t === 'stt.catalog.request') {
      void this.sendSttCatalog();
      return;
    }
    if (msg.t === 'sessions.list.request') {
      void this.sendRecentSessions('list');
      return;
    }
    if (msg.t === 'sessions.catalog.request') {
      void this.sendRecentSessions('catalog');
      return;
    }
    if (msg.t === 'sessions.list.subscribe') {
      this.startRecentSessionsSubscription();
      void this.sendRecentSessions('list');
      return;
    }
    if (msg.t === 'sessions.list.unsubscribe') {
      this.stopRecentSessionsSubscription();
      return;
    }
    if (msg.t === 'sessions.destinations.request') {
      void this.sendNewSessionDestinations();
      return;
    }
    if (msg.t === 'sessions.create.request') {
      void this.sendNewSessionCreateResponse(msg);
      return;
    }
    if (msg.t === 'session.preview') {
      void this.speakSessionPreview(msg.sessionId);
      return;
    }
    if (msg.t === 'stt.start') {
      // Mic priority: a PTT press means the human is about to talk — cut
      // any playing notification (radio, song) immediately (2026-09-17,
      // after field-testing showed voice arriving over the stream).
      this.stopPush();
      // The client kills playback on press (client-side mic priority,
      // 2026-09-18) — the audio it was draining no longer matters.
      this.audioDrainUntilMs = 0;
      this.resetTurn('stt_restart');
      // Routing is room-bound — ignore any payload on stt.start.
      const token = this.beginTurn();
      void this.openStt(token);
      return;
    }
    if (msg.t === 'stt.audio.done') {
      /* Debug: convert PCM dump to WAV */
      if (this._pcmDumpPath && this._pcmDumpBytes > 0) {
        const fs = await import('node:fs');
        const pcm = fs.readFileSync(this._pcmDumpPath);
        const wavHeader = Buffer.alloc(44);
        wavHeader.write('RIFF', 0);
        wavHeader.writeUInt32LE(36 + pcm.length, 4);
        wavHeader.write('WAVE', 8);
        wavHeader.write('fmt ', 12);
        wavHeader.writeUInt32LE(16, 16);
        wavHeader.writeUInt16LE(1, 20);
        wavHeader.writeUInt16LE(1, 22);
        wavHeader.writeUInt32LE(16000, 24);
        wavHeader.writeUInt32LE(32000, 28);
        wavHeader.writeUInt16LE(2, 32);
        wavHeader.writeUInt16LE(16, 34);
        wavHeader.write('data', 36);
        wavHeader.writeUInt32LE(pcm.length, 40);
        const wavPath = this._pcmDumpPath.replace('.pcm', '.wav');
        fs.writeFileSync(wavPath, Buffer.concat([wavHeader, pcm]));
        console.log(`[voice ${this.roomId}] DEBUG: WAV written ${wavPath} (${pcm.length} PCM bytes, ${pcm.length / 32000}s)`);
        this._pcmDumpPath = undefined;
        this._pcmDumpBytes = 0;
      }
      this.stt?.signalAudioDone();
      return;
    }
    if (msg.t === 'stt.cancel') {
      this.resetTurn('stt_cancelled');
      return;
    }
    if (msg.t === 'reply.cancel') {
      this.resetTurn('reply_cancelled');
      return;
    }
    if (msg.t === 'settings.update') {
      this.applyVoiceSettings(msg.settings);
      return;
    }
    if (msg.t === 'push.stop') {
      this.stopPush();
      return;
    }
  }

  private async sendTtsCatalog(): Promise<void> {
    try {
      const loadCatalog = this.opts.ttsCatalogProvider ?? (() => defaultTtsCatalogCache.get());
      const catalog = await loadCatalogWithTimeout(loadCatalog, TTS_CATALOG_LOAD_TIMEOUT_MS);
      this.send(daemonToPhone.ttsCatalog(catalog ?? createEmptyTtsCatalog()));
    } catch (err) {
      console.error(`[voice ${this.roomId}] TTS catalog load failed: ${sanitizedErrorMessage(err)}`);
      this.send(daemonToPhone.ttsCatalog(createEmptyTtsCatalog()));
    }
  }

  private async sendSttCatalog(): Promise<void> {
    try {
      const loadCatalog = this.opts.sttCatalogProvider ?? (() => defaultSttCatalogCache.get());
      const catalog = await loadCatalog();
      this.send(daemonToPhone.sttCatalog(catalog));
    } catch {
      this.send(daemonToPhone.sttCatalog(createEmptySttCatalog()));
    }
  }

  private startRecentSessionsSubscription(): void {
    if (this.recentSessionsInterval) return;
    this.recentSessionsInterval = setInterval(() => {
      void this.sendRecentSessions('list');
    }, RECENT_SESSIONS_SUBSCRIPTION_INTERVAL_MS);
    this.recentSessionsInterval.unref?.();
  }

  private stopRecentSessionsSubscription(): void {
    if (!this.recentSessionsInterval) return;
    clearInterval(this.recentSessionsInterval);
    this.recentSessionsInterval = null;
  }

  // Speak a session's display label as TTS without running a chat turn.
  // Used by screenless clients (the Pi walkie-talkie) to voice the session
  // they stepped onto with switch blips. Skips while a live turn is in
  // flight; a new preview replaces an unfinished one (turn token bump).
  private async speakSessionPreview(sessionId: string): Promise<void> {
    try {
      if (this.state.turnInFlight && !this.tts) return;
      const loadSessions = this.opts.recentSessionsProvider ?? (() => defaultRecentSessionsCache.get());
      const snapshot = await loadSessions();
      const label = (snapshot.sessions ?? []).find((s) => s.sessionId === sessionId)?.displayLabel?.trim();
      if (!label) return;
      if (this.state.turnInFlight && !this.tts) return; // re-check after await
      console.log(`[voice ${this.roomId}] session preview: ${label}`);
      const token = this.beginTurn();
      void this.openTtsAsync(label, token);
    } catch (err) {
      console.error(`[voice ${this.roomId}] session preview failed: ${sanitizedErrorMessage(err)}`);
    }
  }

  private async sendRecentSessions(format: 'list' | 'catalog' = 'list'): Promise<void> {
    const toMessage = format === 'catalog' ? daemonToPhone.sessionsCatalog : daemonToPhone.sessionsList;
    try {
      const loadSessions = this.opts.recentSessionsProvider ?? (() => defaultRecentSessionsCache.get());
      this.send(toMessage(await loadSessions()));
    } catch {
      this.send(toMessage(createEmptyRecentSessionsSnapshot()));
    }
  }

  private async sendNewSessionDestinations(): Promise<void> {
    const immediateCatalog = createWebchatOnlyNewSessionDestinationsCatalog();
    this.send(daemonToPhone.sessionsDestinations(immediateCatalog));
    try {
      const loadCatalog = this.opts.newSessionDestinationsProvider
        ?? (() => getNewSessionDestinationsWithOpenClaw({
          ...(this.opts.newSessionDiscordDestinationsProvider
            ? { loadDiscordDestinations: this.opts.newSessionDiscordDestinationsProvider }
            : {}),
          ...(this.opts.newSessionSlackDestinationsProvider
            ? { loadSlackDestinations: this.opts.newSessionSlackDestinationsProvider }
            : {}),
        }));
      const catalog = await loadCatalog();
      if (!sameNewSessionDestinationProviders(immediateCatalog, catalog)) {
        this.send(daemonToPhone.sessionsDestinations(catalog));
      }
    } catch {
      // The immediate webchat catalog has already kept local sessions usable.
    }
  }

  private async sendNewSessionCreateResponse(msg: NewSessionCreateRequestLike): Promise<void> {
    const respond = this.opts.newSessionCreateResponder ?? buildNewSessionCreateResponse;
    try {
      this.send(await respond(msg));
    } catch {
      this.send(daemonToPhone.sessionsCreateError(
        typeof msg.requestId === 'string' ? msg.requestId : '',
        'new_session_create_failed',
      ));
    }
  }

  private async openStt(token: number): Promise<void> {
    this.sttOpenToken = token;
    console.error(`[voice ${this.roomId}] opening OpenClaw infer STT session`);
    const createStt = this.opts.sttSessionFactory ?? ((opts, cb) => new OpenClawInferSttSession(opts, cb));
    const createSpeechDetector = this.opts.createSpeechDetector ?? createWasmVad;

    // Buffer audio chunks that arrive while the STT session is being created.
    // The web client waits for stt.ready before sending audio; the ESP32 sends
    // immediately after stt.start, so we need to buffer until the session is ready.
    const pendingAudio: Uint8Array[] = [];
    let sessionReady = false;
    this.stt = {
      sendAudio: (bytes) => {
        if (sessionReady) {
          realSession.sendAudio(bytes);
        } else {
          pendingAudio.push(bytes);
        }
      },
      signalAudioDone: () => {
        if (sessionReady) realSession.signalAudioDone();
      },
      close: () => {
        try { realSession.close(); } catch { /* best effort */ }
      },
    };
    let realSession: SttSessionLike;
    let speechDetector: SpeechDetector | undefined;
    let detectorDestroyed = false;
    const destroySpeechDetector = () => {
      if (detectorDestroyed) return;
      detectorDestroyed = true;
      try {
        speechDetector?.destroy?.();
      } catch {
        // best effort cleanup
      }
    };

    try {
      speechDetector = await createSpeechDetector({ sampleRate: STT_SAMPLE_RATE });
    } catch (err) {
      if (!this.isTurnActive(token)) return;
      console.error(`[voice ${this.roomId}] WASM VAD init failed; continuing without phrase chunks: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (!this.isTurnActive(token)) {
      destroySpeechDetector();
      return;
    }

    const enablePhraseChunks = !!speechDetector;
    const sttModel = sttModelOverride(this.sttSelection);
    const sttOptions: OpenClawInferSttSessionOptions = {
      language: this.opts.sttLanguage,
      sampleRate: STT_SAMPLE_RATE,
      enablePhraseChunks,
      ...(sttModel ? { model: sttModel } : {}),
    };
    if (speechDetector) sttOptions.speechDetector = speechDetector;

    let sessionRef: SttSessionLike;
    const newSession = createStt(
      sttOptions,
      {
        onReady: () => {
          if (!this.isTurnActive(token)) return;
          // Flush buffered audio before signaling ready
          // Defer to next tick so `newSession` assignment completes first
          sessionReady = true;
          queueMicrotask(() => {
            for (const buf of pendingAudio) sessionRef.sendAudio(buf);
            pendingAudio.length = 0;
          });
          this.send(daemonToPhone.sttReady());
        },
        onPartial: (text, isFinal) => {
          if (!this.isTurnActive(token)) return;
          this.send(daemonToPhone.sttPartial(text, isFinal));
        },
        onDone: (text) => {
          destroySpeechDetector();
          if (!this.isTurnActive(token)) return;
          this.send(daemonToPhone.sttDone(text));
          this.stt = null;
          void this.runReplyTurn(text, token);
        },
        onError: (message) => {
          destroySpeechDetector();
          if (!this.isTurnActive(token)) return;
          this.send(daemonToPhone.sttError(sanitizeReplyFailureLogText(message)));
          this.stt = null;
          this.resetTurn('stt_error');
        },
        onClosed: () => {
          destroySpeechDetector();
          if (!this.isTurnActive(token)) return;
          this.send(daemonToPhone.sttClosed());
        },
      },
    );
    realSession = newSession;
    sessionRef = newSession;
    // this.stt was already set at the start of openStt with a buffering wrapper.
    // No need to reassign here — the buffer flush happens in onReady.
  }

  private async runReplyTurn(transcript: string, token: number): Promise<void> {
    console.log(`[voice ${this.roomId}] DEBUG: runReplyTurn transcript="${transcript.trim().substring(0, 100)}" token=${token}`);
    if (!this.isTurnActive(token)) return;
    const trimmed = transcript.trim();
    // Volume-intent detector (decision-model exploration, 2026-10-16):
    // intercept volume requests before the agent turn — apply the gain
    // in-process, give a short spoken ack, skip the chat turn entirely.
    // On miss or detector failure the turn proceeds normally.
    if (trimmed) {
      try {
        const { evaluateVolumeIntent } = await import('./volumeIntent.js');
        const intent = await evaluateVolumeIntent(trimmed);
        if (intent?.requested) {
          let appliedDb: number;
          if (intent.absoluteDb !== null) {
            appliedDb = this.applyTtsVolume(intent.absoluteDb);
          } else if (intent.adjustDb !== null) {
            appliedDb = this.adjustTtsVolume(intent.adjustDb);
          } else {
            appliedDb = this.adjustTtsVolume(3);
          }
          const note = intent.absoluteDb !== null ? appliedDb : (appliedDb - this.ttsGainDb);
          if (intent.absoluteDb !== null) console.log(`[voice ${this.roomId}] volume absolute ${intent.absoluteDb} applied=${appliedDb}`);
          else console.log(`[voice ${this.roomId}] volume adjust ${intent.adjustDb} applied=${appliedDb}`);
          // Spoken ack (short, TTS) — same replyDone+TTS path as chat turns,
          // minus the chat turn. The board reads replyDone for display and
          // plays the tiny ack.
          const ackText = intent.absoluteDb !== null
            ? (intent.absoluteDb >= 12 ? "Max volume." : (intent.absoluteDb <= -20 ? "Muted." : `Volume set to ${intent.absoluteDb > 0 ? '+' : ''}${intent.absoluteDb} decibels.`))
            : (intent.adjustDb !== null && intent.adjustDb > 0 ? "Louder." : (intent.adjustDb !== null && intent.adjustDb < 0 ? "Quieter." : "Volume adjusted."));
          this.send(daemonToPhone.replyDone(ackText));
          void this.openTtsAsync(ackText, token);
          this.resetTurn('volume_intent');
          return;
        }
      } catch (err) {
        console.warn(`[voice ${this.roomId}] volume-intent detector error (ignoring): ${err instanceof Error ? err.message : err}`);
        // fall through — chat turn runs
      }
    }
    if (!trimmed) {
      console.log(`[voice ${this.roomId}] DEBUG: empty transcript, sending error`);
      this.send(daemonToPhone.replyError('empty_transcript'));
      this.resetTurn('empty_transcript');
      return;
    }
    this.send(daemonToPhone.replyStart(trimmed));
    this.chatAbort = new AbortController();
    let replyText: string;
    console.log(`[voice ${this.roomId}] DEBUG: runChat starting transcript="${trimmed.substring(0, 80)}"`);
    try {
      const target = this.state.chatTarget();
      const result = await runChat(trimmed, {
        signal: this.chatAbort.signal,
        sessionId: target.sessionId,
        originRoomId: this.roomId,
        ...(target.sessionKey ? { sessionKey: target.sessionKey } : {}),
        ...(target.channel ? { channel: target.channel } : {}),
        ...(target.target ? { target: target.target } : {}),
        ...(target.accountId ? { accountId: target.accountId } : {}),
        delivery: target.delivery,
        deliver: shouldDeliverReplyForChatTarget(target),
      });
      if (!this.isTurnActive(token)) return;
      replyText = result.text;
    } catch (err) {
      this.chatAbort = null;
      if (!this.isTurnActive(token)) return;
      const code = err instanceof ChatError ? err.code : 'reply_failed';
      this.logReplyFailure(code, err);
      this.send(daemonToPhone.replyError(code));
      this.resetTurn('reply_error');
      return;
    }
    this.chatAbort = null;
    if (!this.isTurnActive(token)) return;
    console.log(`[voice ${this.roomId}] DEBUG: runChat done reply="${replyText.substring(0, 80)}" opening TTS`);
    this.send(daemonToPhone.replyDone(replyText));
    void this.openTtsAsync(replyText, token);
  }


  private logReplyFailure(code: string, err: unknown): void {
    const details = err && typeof err === 'object'
      ? (err as { details?: { rootMessage?: string; stderr?: string; exitCode?: string } }).details
      : undefined;
    const rootMessage = sanitizeReplyFailureLogText(
      details?.rootMessage ?? (err instanceof Error ? err.message : 'unknown'),
    );
    const stderr = details?.stderr ? sanitizeReplyFailureLogText(details.stderr) : undefined;
    const exitCode = details?.exitCode;
    const target = this.state.chatTarget();
    const delivery = target.delivery ? `${target.delivery.channel}:${target.delivery.target}` : 'explicit-reply-target-unresolved';
    const fields = [
      `[voice ${this.roomId}] reply failed`,
      `session=${sanitizeReplyFailureLogText(target.sessionId)}`,
      target.sessionKey ? `sessionKey=${sanitizeReplyFailureLogText(target.sessionKey)}` : undefined,
      target.channel ? `channel=${sanitizeReplyFailureLogText(target.channel)}` : undefined,
      target.target ? `target=${sanitizeReplyFailureLogText(target.target)}` : undefined,
      target.accountId ? `accountId=${sanitizeReplyFailureLogText(target.accountId)}` : undefined,
      `delivery=${sanitizeReplyFailureLogText(delivery)}`,
      `code=${sanitizeReplyFailureLogText(code)}`,
      exitCode ? `exit=${sanitizeReplyFailureLogText(exitCode)}` : undefined,
      `message=${rootMessage}`,
      stderr ? `stderr=${stderr}` : undefined,
    ].filter((field): field is string => Boolean(field));
    console.error(fields.join(' '));
  }

  private startTtsAudioTurn(token: number, text: string): void {
    this.ttsAudioTurn = {
      turnId: token,
      text,
      sampleRate: TTS_SAMPLE_RATE,
      chunks: [],
      byteLength: 0,
      started: false,
      abandoned: false,
      complete: false,
      drained: false,
      overflowed: false,
      replayOnReconnect: false,
      replayFromChunkIndex: 0,
      liveDataCursor: 0,
    };
  }

  private markTtsAudioReplayOnReconnect(token?: number, fromChunkIndex?: number): void {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.drained || turn.overflowed || turn.abandoned) return;
    if (token !== undefined && turn.turnId !== token) return;
    const replayStart = Math.max(0, Math.min(fromChunkIndex ?? 0, turn.chunks.length));
    turn.replayFromChunkIndex = turn.replayOnReconnect
      ? Math.min(turn.replayFromChunkIndex, replayStart)
      : replayStart;
    turn.replayOnReconnect = true;
  }

  private markTtsAudioStarted(token?: number): void {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.abandoned) return;
    if (token !== undefined && turn.turnId !== token) return;
    turn.started = true;
  }

  private abandonTtsAudioTurn(token?: number): void {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.drained) return;
    if (token !== undefined && turn.turnId !== token) return;
    turn.abandoned = true;
    turn.replayOnReconnect = false;
    turn.replayFromChunkIndex = 0;
  }

  private appendTtsAudioPcm(token: number, pcm: Uint8Array): { chunk: Buffer; retained: boolean } | null {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.turnId !== token || turn.drained || pcm.byteLength === 0) return null;
    const chunk = Buffer.from(pcm);
    if (turn.overflowed) return { chunk, retained: false };
    if (turn.byteLength + chunk.byteLength > MAX_BUFFERED_TTS_TURN_BYTES) {
      // The cap is only a reconnect-replay retention cap. Once exceeded, drop
      // the retained canonical replay log for this turn, but keep returning the
      // current/future live chunks so connected transports are not cut off.
      turn.chunks = [];
      turn.byteLength = 0;
      turn.overflowed = true;
      turn.replayOnReconnect = false;
      turn.replayFromChunkIndex = 0;
      turn.liveDataCursor = 0;
      console.error(`[voice ${this.roomId}] buffered TTS replay dropped: canonical queue exceeded ${MAX_BUFFERED_TTS_TURN_BYTES} bytes`);
      return { chunk, retained: false };
    }
    turn.chunks.push(chunk);
    turn.byteLength += chunk.byteLength;
    return { chunk, retained: true };
  }

  /**
   * Realtime pacing pump for data-channel TTS (rewritten 2026-10-08). The
   * ESP32 speaker ring is 2s and its recv callback blocks the peer loop when
   * full, so chunks leave only as the byte budget allows: the budget grows
   * at realtime per tick (capped ~1s) and a chunk is sent only when the
   * budget covers it. Generation faster than playback can no longer overrun
   * the board. v1's counter-based gate never counted newly-arrived chunks
   * and blocked forever — never again count what arrives; count what may
   * LEAVE.
   */
  private ensureTtsPacePump(token: number): void {
    if (this.ttsPaceTimer) {
      if (this.ttsPaceToken === token) return;
      this.stopTtsPacePump();
    }
    this.ttsPaceToken = token;
    const turn = this.ttsAudioTurn;
    if (turn) this.ttsPaceCursor = turn.liveDataCursor;
    // Head-start (11:19 rewrite): prime the receiver's buffer with ~1.5s of
    // audio immediately so ticks only SUSTAIN depth (overdrive 1.15x), not
    // build it from zero — 1.0x-with-zero-slack was the 11:16 field pause.
    this.ttsPaceBudget = TTS_PACE_HEADSTART_BYTES;
    this.ttsPaceLastGrowAt = 0;
    // Diag reset (field timing instrumentation, live pump→send path)
    this.ttsDiagStartedAt = Date.now();
    this.ttsDiagSentCount = 0;
    this.ttsDiagLastSendAt = this.ttsDiagStartedAt;
    this.ttsDiagLastTickAt = this.ttsDiagStartedAt;
    this.ttsDiagStarvedLogged = false;
    console.log(`[voice ${this.roomId}] [tts-diag] pump start t=${this.ttsDiagStartedAt}`);
    this.ttsPaceTimer = setInterval(() => {
      const now = Date.now();
      const tickDelay = this.ttsDiagLastTickAt ? now - this.ttsDiagLastTickAt : 0;
      this.ttsDiagLastTickAt = now;
      // Event-loop stall: ticks were scheduled every TTS_PACE_TICK_MS; big
      // delays here mean the NODE EVENT LOOP was blocked (not the transport).
      if (tickDelay > TTS_PACE_TICK_MS * 5) {
        const backlogNow = (this.ttsAudioTurn?.chunks.length ?? 0) - this.ttsPaceCursor;
        console.warn(`[voice ${this.roomId}] [tts-diag] EVENT LOOP STALL ${tickDelay}ms between ticks (backlog ${backlogNow}, budget ${this.ttsPaceBudget})`);
      }
      if (this.ttsPaceToken !== token || !this.isTurnActive(token)) {
        this.stopTtsPacePump();
        return;
      }
      const t = this.ttsAudioTurn;
      if (!t || t.turnId !== token || t.drained || t.abandoned) {
        this.stopTtsPacePump();
        if (this.ttsPaceDonePending) {
          this.ttsPaceDonePending = false;
          this.completeTtsTurn('tts_pace_transport');
        }
        return;
      }
      // Grow budget at realtime × overdrive, capped at lookahead.
      // TIME-BASED (field 2026-10-15 08:23): the per-tick constant assumed
      // 40ms ticks; a busy event loop stretches ticks to 80–110ms, so the
      // board was fed at ~0.6x realtime (good-morning turn: 718KB/40s) —
      // the stutter root cause after the pump-logging fix. Grow by ACTUAL
      // elapsed time; the lookahead cap still bounds SCTP backlog.
      const growDtMs = this.ttsPaceLastGrowAt ? Math.max(0, now - this.ttsPaceLastGrowAt) : TTS_PACE_TICK_MS;
      this.ttsPaceLastGrowAt = now;
      this.ttsPaceBudget = Math.min(
        this.ttsPaceBudget + (TTS_PACE_BYTES_PER_TICK * TTS_PACE_RATE * growDtMs) / TTS_PACE_TICK_MS,
        TTS_PACE_LOOKAHEAD_BYTES,
      );
      while (this.ttsPaceCursor < t.chunks.length) {
        // Backpressure check against SCTP send buffer to prevent RTO congestion collapse
        const rawChannel = (this.peer as any)?._channel;
        const buffered = typeof rawChannel?.bufferedAmount === 'number' ? rawChannel.bufferedAmount : 0;
        if (buffered >= MAX_SCTP_BUFFERED_AMOUNT) {
          break; // Let SCTP drain and ACK before stuffing more chunks
        }
        const chunk = t.chunks[this.ttsPaceCursor];
        if (chunk.byteLength > this.ttsPaceBudget) {
          // Starved: backlog exists but budget too small TO SEND (not an
          // error in itself — log once per turn at severity).
          if (!this.ttsDiagStarvedLogged) {
            console.warn(`[voice ${this.roomId}] [tts-diag] starved tick: backlog=${t.chunks.length - this.ttsPaceCursor} budget=${this.ttsPaceBudget} chunk=${chunk.byteLength}`);
            this.ttsDiagStarvedLogged = true;
          }
          break;
        }
        const chunkToSend = this.sampleRateForTts === STT_SAMPLE_RATE
          ? chunk
          : resamplePcm(chunk, TTS_SAMPLE_RATE, STT_SAMPLE_RATE);
        if (!this.sendBinary(new Uint8Array(chunkToSend))) {
          console.error(`[voice ${this.roomId}] [tts-diag] sendBinary FAILED chunk #${this.ttsPaceCursor}`);
          if (t.started) {
            this.abandonTtsAudioTurn(token);
          } else {
            this.markTtsAudioReplayOnReconnect(token, this.ttsPaceCursor);
          }
          this.stopTtsPacePump();
          if (this.ttsPaceDonePending) {
            this.ttsPaceDonePending = false;
            this.completeTtsTurn('tts_pace_transport');
          }
          return;
        }
        this.markTtsAudioStarted(token);
        this.ttsPaceBudget -= chunk.byteLength;
        this.ttsPaceCursor += 1;
        const sendAt = Date.now();
        const sendGap = sendAt - this.ttsDiagLastSendAt;
        this.ttsDiagLastSendAt = sendAt;
        this.ttsDiagSentCount += 1;
        // Gap between sends > 3s: the FIELD stall signature — name whether
        // backlog was available (transport/board slow) or empty (starved).
        if (sendGap > 3000) {
          console.warn(`[voice ${this.roomId}] [tts-diag] SEND GAP ${sendGap}ms before chunk #${this.ttsDiagSentCount} (backlog after=${t.chunks.length - this.ttsPaceCursor}, budget after=${this.ttsPaceBudget})`);
        }
        t.liveDataCursor = this.ttsPaceCursor; // keep reconnect-replay cursor in sync
      }
      if (this.ttsPaceCursor >= t.chunks.length && t.complete) {
        this.stopTtsPacePump();
        console.log(`[voice ${this.roomId}] [tts-diag] pump drain complete: sent=${this.ttsDiagSentCount} elapsed=${Date.now() - (this.ttsDiagStartedAt ?? Date.now())}ms`);
        if (this.ttsPaceDonePending) {
          this.ttsPaceDonePending = false;
          this.completeTtsTurn('tts_pace_done');
        }
      }
      // otherwise: timer idles until more chunks arrive or generation completes
    }, TTS_PACE_TICK_MS);
    this.ttsPaceTimer.unref?.();
  }

  private stopTtsPacePump(): void {
    if (this.ttsPaceTimer) {
      clearInterval(this.ttsPaceTimer);
      this.ttsPaceTimer = null;
    }
    this.ttsPaceToken = null;
    this.ttsPaceCursor = 0;
    this.ttsPaceBudget = 0;
  }

  private drainLiveDataTtsAudio(token: number): void {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.turnId !== token || turn.drained || turn.overflowed || turn.abandoned) return;
    if (turn.replayOnReconnect) return;
    while (turn.liveDataCursor < turn.chunks.length) {
      const chunk = turn.chunks[turn.liveDataCursor];
      // Resample only when TTS output is not already at the data channel rate
      const chunkToSend = this.sampleRateForTts === STT_SAMPLE_RATE
        ? chunk
        : resamplePcm(chunk, TTS_SAMPLE_RATE, STT_SAMPLE_RATE);
      if (!this.sendBinary(new Uint8Array(chunkToSend))) {
        if (turn.started) {
          this.abandonTtsAudioTurn(token);
        } else {
          this.markTtsAudioReplayOnReconnect(token, turn.liveDataCursor);
        }
        return;
      }
      this.markTtsAudioStarted(token);
      turn.liveDataCursor += 1;
    }
  }

  private markTtsAudioComplete(token: number): void {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.turnId !== token || turn.overflowed) return;
    turn.complete = true;
  }

  private clearTtsAudioTurn(): void {
    this.ttsAudioTurn = null;
  }

  private completeTtsTurn(reason: string): void {
    // Paced data-channel turns: never send ttsDone while the pace pump still
    // has undelivered canonical chunks — defer and let the pump complete.
    const pendingTurn = this.ttsAudioTurn;
    if (
      this.ttsPaceTimer &&
      this.ttsPaceToken === this.turnToken &&
      pendingTurn && pendingTurn.turnId === this.turnToken &&
      !pendingTurn.abandoned &&
      this.ttsPaceCursor < pendingTurn.chunks.length
    ) {
      this.ttsPaceDonePending = true;
      return;
    }
    this.ttsPaceDonePending = false;
    this.markTtsAudioComplete(this.turnToken);
    const audioTurn = this.ttsAudioTurn;
    if (
      audioTurn &&
      audioTurn.turnId === this.turnToken &&
      (audioTurn.abandoned || !audioTurn.replayOnReconnect || audioTurn.overflowed || audioTurn.replayFromChunkIndex >= audioTurn.chunks.length)
    ) {
      this.clearTtsAudioTurn();
    }
    this.audioPumpAwaitingDone = false;
    this.send(daemonToPhone.ttsDone());
    if (reason !== 'tts_audio_transport_closed' && this.connected && this.peer && !this.peer.destroyed) {
      this.drainTtsAudioReplay(this.peer);
    }
    // Switchboard: deliver a queued line switch now that the reply finished —
    // the phone defers until its playback buffer drains, then re-homes.
    if (this.pendingSessionSwitch && this.connected && this.peer && !this.peer.destroyed) {
      const sw = this.pendingSessionSwitch;
      this.pendingSessionSwitch = null;
      console.log(`[voice ${this.roomId}] Delivering queued session switch → ${sw.session}`);
      this.sendToPeer(this.peer, daemonToPhone.sessionSwitch(sw.session, sw.sessionKey));
    }
    if (this.pendingLineSwitch && this.connected && this.peer && !this.peer.destroyed) {
      const line = this.pendingLineSwitch;
      this.pendingLineSwitch = null;
      this.sendToPeer(this.peer, daemonToPhone.lineSwitch(line));
    }
    this.tts = null;
    this.resetTurn(reason);
  }

  private drainTtsAudioReplay(peer: SimplePeer.Instance): void {
    const turn = this.ttsAudioTurn;
    if (!turn || turn.drained || !turn.complete || turn.overflowed) return;
    if (turn.abandoned) {
      this.clearTtsAudioTurn();
      return;
    }
    if (!turn.replayOnReconnect) return;
    const chunks = turn.chunks.slice(turn.replayFromChunkIndex);
    if (chunks.length === 0) {
      turn.drained = true;
      this.clearTtsAudioTurn();
      return;
    }
    if (!this.sendToPeer(peer, daemonToPhone.ttsStart(turn.sampleRate, {
      buffered: true,
      turnId: turn.turnId,
      text: turn.text,
    }))) return;
    for (const chunk of chunks) {
      if (!this.sendBinaryToPeer(peer, chunk, 'buffered binary send failed')) return;
    }
    if (!this.sendToPeer(peer, daemonToPhone.ttsDone())) return;
    turn.drained = true;
    this.clearTtsAudioTurn();
  }

  private async openTtsAsync(text: string, token: number): Promise<void> {
    try {
      this.audioFrameQueue = [];
      this.rawRemainder = Buffer.alloc(0);
      this.resampledRemainder = Buffer.alloc(0);
      this.audioPumpAwaitingDone = false;
      this.startTtsAudioTurn(token, text);
      const catalog = await this.loadTtsCatalogForTurn();
      if (!this.isTurnActive(token)) return;
      const useTrack = !!this.audioSource;
      // Streaming TTS (experimental/tts-stream): pipe ElevenLabs audio as
      // it is generated — first audio in ~1s instead of after the whole MP3
      // is synthesized (~28s on long replies). If the stream fails before
      // any audio (spawn/auth error), fall back to the whole-file infer path.
      const makeStreamingTts = (opts: TtsSessionOptions, cb: TtsSessionCallbacks) => {
        let gotAudio = false;
        return new ElevenLabsTtsSession({
          text: opts.text,
          ...(opts.voice ? { voiceId: opts.voice } : {}),
          ...(opts.model ? { modelId: opts.model } : {}),
        }, {
          onOpen: cb.onOpen,
          onAudio: (pcm) => { gotAudio = true; cb.onAudio(pcm); },
          onDone: cb.onDone,
          onError: (message) => {
            if (!gotAudio && this.isTurnActive(token)) {
              console.warn(`[voice ${this.roomId}] TTS stream failed before audio (${message}) — falling back to infer convert`);
              try { new OpenClawInferTtsSession(opts, cb); } catch { cb.onError(message); }
            } else {
              cb.onError(message);
            }
          },
        });
      };
      const createTts = this.opts.ttsSessionFactory
        ?? (process.env.ELEVENLABS_API_KEY
          ? makeStreamingTts
          : ((opts, cb) => new OpenClawInferTtsSession(opts, cb)));
      const request = buildTtsSessionRequest(text, this.ttsSelection, { catalog });
      // When sending via data channel (not WebRTC audio track), request 16kHz PCM
      // directly from the TTS engine to skip the 24kHz→16kHz resample step.
      if (!useTrack) {
        request.sampleRate = STT_SAMPLE_RATE;
        this.sampleRateForTts = STT_SAMPLE_RATE;
      } else {
        this.sampleRateForTts = TTS_SAMPLE_RATE;
      }
      this.tts = createTts(request, {
        onOpen: () => {
          if (!this.isTurnActive(token)) return;
          this.send(daemonToPhone.ttsStart(useTrack ? WEBRTC_SAMPLE_RATE : STT_SAMPLE_RATE));
        },
        onAudio: (pcm) => {
          if (!this.isTurnActive(token)) return;
          const audio = this.appendTtsAudioPcm(token, this.gainPcm(pcm));
          if (!audio) return;
          if (useTrack) {
            const turn = this.ttsAudioTurn;
            if (this.connected && this.audioSource && turn && !turn.abandoned && (!turn.replayOnReconnect || turn.overflowed)) {
              this.enqueueTtsPcm(audio.chunk);
            } else if (audio.retained && !turn?.abandoned) {
              this.markTtsAudioReplayOnReconnect(token, 0);
            }
            return;
          }
          // Direct delivery: send chunk straight to the data channel as it arrives from ElevenLabs.
          if (this.sampleRateForTts === STT_SAMPLE_RATE) {
            this.sendBinary(new Uint8Array(audio.chunk));
          } else {
            const resampled = resamplePcm(Buffer.from(audio.chunk), TTS_SAMPLE_RATE, STT_SAMPLE_RATE);
            this.sendBinary(new Uint8Array(resampled));
          }
        },
        onDone: () => {
          if (!this.isTurnActive(token)) return;
          this.markTtsAudioComplete(token);
          if (!useTrack || !this.audioSource) {
            this.completeTtsTurn('tts_done');
            return;
          }
          this.flushTtsTail(); this.audioPumpAwaitingDone = true; this.startAudioPump(token);
          if (this.audioFrameQueue.length === 0) {
            this.completeTtsTurn('tts_done');
          }
        },
        onError: (message) => {
          if (!this.isTurnActive(token)) return;
          this.clearTtsAudioTurn();
          this.stopAudioPump();
          this.stopTtsPacePump();
          this.ttsPaceDonePending = false;
          this.audioFrameQueue = [];
          this.rawRemainder = Buffer.alloc(0);
          this.resampledRemainder = Buffer.alloc(0);
          this.audioPumpAwaitingDone = false;
          this.send(daemonToPhone.ttsError(sanitizeReplyFailureLogText(message)));
          this.tts = null;
          this.resetTurn('tts_error');
        },
      });
    } catch (err) {
      if (!this.isTurnActive(token)) return;
      this.clearTtsAudioTurn();
      console.error(`[voice ${this.roomId}] TTS open failed: ${sanitizedErrorMessage(err)}`);
      this.send(daemonToPhone.ttsError('openclaw_infer_tts_failed'));
      this.resetTurn('tts_open_failed');
    }
  }

  private async loadTtsCatalogForTurn(): Promise<TtsCatalog | null> {
    const loadCatalog = this.opts.ttsCatalogProvider ?? (() => defaultTtsCatalogCache.get());
    try { return await loadCatalogWithTimeout(loadCatalog, TTS_CATALOG_LOAD_TIMEOUT_MS); }
    catch (err) { console.error(`[voice ${this.roomId}] TTS catalog load failed: ${sanitizedErrorMessage(err)}`); return null; }
  }


  private beginTurn(): number {
    this.touchActivity();
    this.clearTtsAudioTurn();
    const token = ++this.turnToken;
    this.sttOpenToken = token;
    this.state.handleStartTurn();
    this.turnSnapshot = {
      inFlight: true,
      phase: 'recording',
      latestEventId: this.controlEventId,
    };
    return token;
  }

  private isTurnActive(token: number): boolean {
    return token === this.turnToken && this.state.turnInFlight && !this.closing && !this.protocolUnsupported;
  }

  private enterProtocolUnsupported(): void {
    this.protocolUnsupported = true;
    this.stopRecentSessionsSubscription();
    this.resetTurn('protocol_unsupported');
  }

  private resetTurn(reason: string): void {
    this.touchActivity();
    if (!['tts_done', 'tts_audio_pump_done', 'tts_audio_transport_closed'].includes(reason)) {
      this.clearTtsAudioTurn();
    }
    this.turnToken += 1;
    this.sttOpenToken = this.turnToken;
    this.state.resetTurn();
    try {
      this.stt?.close();
    } catch {
      // ignore
    }
    this.stt = null;
    try {
      this.tts?.cancel();
    } catch {
      // ignore
    }
    this.tts = null;
    this.stopAudioPump();
    this.stopTtsPacePump();
    this.ttsPaceDonePending = false;
    this.audioFrameQueue = [];
    this.rawRemainder = Buffer.alloc(0);
    this.resampledRemainder = Buffer.alloc(0);
    this.audioPumpAwaitingDone = false;
    this.chatAbort?.abort();
    this.chatAbort = null;

    // Switchboard (2026-10-11): resetTurn also runs on ERROR paths (reply_error,
    // empty_transcript, …). completeTtsTurn flushes the queued switches only on
    // the success path, so a turn that died mid-flight stranded the queued
    // switch forever — field case 17:01:54: 'Session switch queued → 669a…'
    // never delivered, phone stuck in Room 2. Flush here for every path that
    // ends a turn; completeTtsTurn's flush runs first on success, so this is
    // the catch-all. Peer/connected guards match the success-path delivery.
    if (this.pendingSessionSwitch && this.connected && this.peer && !this.peer.destroyed) {
      const sw = this.pendingSessionSwitch;
      this.pendingSessionSwitch = null;
      console.log(`[voice ${this.roomId}] Delivering queued session switch (resetTurn/${reason}) → ${sw.session}`);
      this.sendToPeer(this.peer, daemonToPhone.sessionSwitch(sw.session, sw.sessionKey));
    }
    if (this.pendingLineSwitch && this.connected && this.peer && !this.peer.destroyed) {
      const line = this.pendingLineSwitch;
      this.pendingLineSwitch = null;
      console.log(`[voice ${this.roomId}] Delivering queued line switch (resetTurn/${reason}) → ${line}`);
      this.sendToPeer(this.peer, daemonToPhone.lineSwitch(line));
    }
  }

  private send(msg: DaemonToPhone): void {
    if (this.protocolUnsupported) return;
    if (msg.t === 'session.snapshot') return;
    const event = this.recordControlEvent(msg);
    const peer = this.peer;
    if (!peer || !this.connected) return;
    this.sendToPeer(peer, event.msg);
  }

  private sendConnectionScoped(msg: DaemonToPhone): void {
    if (msg.t === 'session.snapshot') return;
    const peer = this.peer;
    if (!peer || !this.connected) return;
    this.sendToPeer(peer, msg);
  }

  private recordControlEvent(msg: DaemonToPhoneEvent): ControlEventRecord {
    const event: ControlEventRecord = { id: ++this.controlEventId, msg };
    this.controlHistory.push(event);
    if (this.controlHistory.length > MAX_CONTROL_HISTORY_EVENTS) {
      this.controlHistory.splice(0, this.controlHistory.length - MAX_CONTROL_HISTORY_EVENTS);
    }
    this.updateTurnSnapshotFromEvent(event);
    return event;
  }

  private updateTurnSnapshotFromEvent(event: ControlEventRecord): void {
    const msg = event.msg;
    const base: VoiceTurnSnapshot = {
      ...this.turnSnapshot,
      inFlight: this.state.turnInFlight,
      latestEventId: event.id,
    };
    if (msg.t === 'stt.done') {
      this.turnSnapshot = { ...base, phase: 'thinking', userText: msg.text, error: undefined };
      return;
    }
    if (msg.t === 'reply.start') {
      this.turnSnapshot = { ...base, phase: 'thinking', userText: msg.text, error: undefined };
      return;
    }
    if (msg.t === 'reply.done') {
      this.turnSnapshot = { ...base, phase: 'reply_ready', replyText: msg.text, error: undefined };
      return;
    }
    if (msg.t === 'tts.start') {
      this.turnSnapshot = { ...base, phase: 'speaking', ttsSampleRate: msg.sample_rate, error: undefined };
      return;
    }
    if (msg.t === 'tts.done') {
      this.turnSnapshot = { ...base, inFlight: false, phase: 'complete', error: undefined };
      return;
    }
    if (msg.t === 'stt.error' || msg.t === 'reply.error' || msg.t === 'tts.error') {
      this.turnSnapshot = { ...base, inFlight: false, phase: 'error', error: msg.message };
      return;
    }
    this.turnSnapshot = base;
  }

  private sendCatchUpSnapshot(peer: SimplePeer.Instance): void {
    const events = this.controlHistory
      .filter((event) => event.id > this.lastPeerDetachEventId)
      .map((event) => ({ id: event.id, msg: event.msg }));
    const disconnectedMs = this.lastPeerDetachAtMs === null
      ? 0
      : Math.max(0, Date.now() - this.lastPeerDetachAtMs);
    this.sendToPeer(peer, daemonToPhone.sessionSnapshot({
      roomId: this.roomId,
      latestEventId: this.controlEventId,
      disconnectedMs,
      turn: { ...this.turnSnapshot, latestEventId: this.controlEventId },
      events,
    }));
    this.drainTtsAudioReplay(peer);
  }

  private sendToPeer(peer: SimplePeer.Instance, msg: unknown): boolean {
    try {
      peer.send(JSON.stringify(msg));
      return true;
    } catch (err) {
      console.error(`[voice ${this.roomId}] send failed: ${err instanceof Error ? err.message : String(err)}`);
      return false;
    }
  }

  private sendBinary(pcm: Uint8Array): boolean {
    this.noteAudioSent(pcm.byteLength);
    if (this.protocolUnsupported) return false;
    const peer = this.peer;
    if (!peer || !this.connected) return false;
    return this.sendBinaryToPeer(peer, pcm, 'binary send failed');
  }

  private sendBinaryToPeer(peer: SimplePeer.Instance, pcm: Uint8Array, label: string): boolean {
    try {
      const copy = new Uint8Array(pcm.byteLength);
      copy.set(pcm);
      peer.send(copy);
      return true;
    } catch (err) {
      console.error(`[voice ${this.roomId}] ${label}: ${err instanceof Error ? err.message : String(err)}`);
      return false;
    }
  }
}


function trimmedString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
}

function objectRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}

function normalizeTtsSelection(settings: VoiceSettings | null | undefined): TtsSelection {
  const settingsRecord = objectRecord(settings);
  const tts = objectRecord(settingsRecord.tts);
  const voice = trimmedString(tts.voice) ?? trimmedString(settingsRecord.voice);
  const providerId = trimmedString(tts.providerId);
  const model = trimmedString(tts.model);
  return {
    ...(providerId ? { providerId } : {}),
    ...(model ? { model } : {}),
    ...(voice ? { voice } : {}),
  };
}

function ttsModelOverride(selection: TtsSelection): string | undefined {
  return selection.providerId && selection.model
    ? `${selection.providerId}/${selection.model}`
    : undefined;
}

export function buildTtsSessionRequest(
  text: string,
  selection: TtsSelection,
  options: { catalog?: TtsCatalog | null } = {},
): { text: string; model?: string; voice?: string; sampleRate?: number } {
  const model = ttsModelOverride(selection);
  const providerId = trimmedString(selection.providerId);
  const voice = trimmedString(selection.voice);
  const forwardVoice = shouldForwardTtsVoice({ providerId, model, voice, catalog: options.catalog });
  return { text, ...(model ? { model } : {}), ...(forwardVoice && voice ? { voice } : {}) };
}

function shouldForwardTtsVoice(input: { providerId?: string; model?: string; voice?: string; catalog?: TtsCatalog | null }): boolean {
  if (!input.voice) return false;
  if (!input.model && !input.providerId) return false;
  const providerId = input.providerId?.toLowerCase();
  const provider = input.catalog?.providers.find((item) => item.id.toLowerCase() === providerId);
  if (provider) {
    if (provider.voices.length === 0) return true;
    return provider.voices.some((candidate) => candidate.id === input.voice);
  }
  if (providerId === OPENAI_PROVIDER_ID && LEGACY_CLAWKIE_TTS_VOICE_IDS.has(input.voice.toLowerCase())) return false;
  return true;
}


function sameNewSessionDestinationProviders(
  left: NewSessionDestinationsCatalog,
  right: NewSessionDestinationsCatalog,
): boolean {
  return JSON.stringify(left.providers) === JSON.stringify(right.providers);
}

function loadCatalogWithTimeout<T>(loadCatalog: () => Promise<T>, timeoutMs: number): Promise<T | null> {
  let timeout: NodeJS.Timeout | undefined;
  const timeoutPromise = new Promise<null>((resolve) => { timeout = setTimeout(() => resolve(null), timeoutMs); timeout.unref?.(); });
  return Promise.race([loadCatalog(), timeoutPromise]).finally(() => { if (timeout) clearTimeout(timeout); });
}


function normalizeSttSelection(settings: VoiceSettings | null | undefined): SttSelection {
  const settingsRecord = objectRecord(settings);
  const stt = objectRecord(settingsRecord.stt);
  const providerId = trimmedString(stt.providerId);
  const model = trimmedString(stt.model);
  return {
    ...(providerId ? { providerId } : {}),
    ...(model ? { model } : {}),
  };
}

function sttModelOverride(selection: SttSelection): string | undefined {
  return selection.providerId && selection.model
    ? `${selection.providerId}/${selection.model}`
    : undefined;
}

function toBytes(data: unknown): Uint8Array | null {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) {
    const view = data as ArrayBufferView;
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  }
  if (typeof data === 'string') {
    return new TextEncoder().encode(data);
  }
  return null;
}

function tryDecodeJsonText(bytes: Uint8Array): string | null {
  if (bytes.length === 0) return null;
  if (bytes[0] !== 0x7b /* { */) return null;
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}
