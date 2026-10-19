/**
 * tailscaleVoiceServer — the daemon half of the WG voice lane.
 *
 * Phase-2 iteration (2026-10-17): boards with microlink (esp32-9b9198 round
 * board, 100.108.153.58) expose a direct, always-dialable UDP lane on their
 * tailnet IP. Wire format on udp/9100 (mirrors the firmware's 1-byte
 * discriminator):
 *   - first byte '{' → clawkie_protocol JSON control (identical message set
 *     to the webrtc data channel)
 *   - anything else  → raw PCM16 @ 16 kHz mono (uplink PCM chunks)
 *
 * The server drives the SAME pipeline as the rendezvous lane:
 * OpenClawInferSttSession → runChat → ElevenLabsTtsSession — protocol
 * identical (stt.ready/partial/done, reply.done, tts.done), transport
 * different. In-home voice no longer needs the signaling server.
 *
 * Source allowlist: only packets from the registered board's VPN IP are
 * accepted (the daemon's LAN is trusted at L2, the lane is WG-encrypted
 * end to end, but board identity is still pinned).
 */

import { createSocket, type RemoteInfo } from 'node:dgram';
import { daemonToPhone } from './protocol.js';
import { OpenClawInferSttSession } from './inferSttSession.js';
// eslint-disable-next-line import/order
import type { SttSessionLike } from './voiceSession.js';
import { ElevenLabsTtsSession } from './elevenLabsTtsSession.js';
import { runChat } from './chatSession.js';
import { defaultRecentSessionsCache, createEmptyRecentSessionsSnapshot } from './recentSessions.js';   /* FIELD 2026-10-19 13:20: session color rides sessions.list — was never sent on the WG lane, board stayed on compiled-in fallback blue */
import type { RecentSessionsSnapshot } from './protocol.js';

const WG_VOICE_PORT = 9100;
// tailscaled (userspace netstack) DROPs unsolicited inbound UDP: local
// sockets only receive peer packets when a matching outbound flow exists
// (symmetric 5-tuple, ~30s idle timeout). The board's voice uplink rides
// board:9100 → VM:9100, so the daemon MUST keep that flow warm — keepalive
// interval 10s, sent from this same socket. Field 2026-10-17 13:10: the
// remote-PTT diag recorded 448 uplink chunks on the board that the daemon
// never saw — pure listener, zero flow, silent drop.
const FLOW_KEEPALIVE_MS = 10_000;
// Tailscale tunnel MTU is 1280: UDP payload must be ≤1252B to avoid
// fragmentation loss (field-probed 2026-10-17: 1000B ✓, 1280B ✗).
// 1200B = 37.5ms @16k mono16, 28B headroom.
const PCM_CHUNK_BYTES = 1200;
const PCM_CHUNK_MS = 37.5;   // 1200B / (16kHz * 2B) — audio duration per chunk
const STT_SAMPLE_RATE = 16000;

/** Registered tailnet boards (deviceId lane → config). */
interface TsBoard {
  deviceId: string;
  vpnIp: string;
  sessionId: string;
  sessionKey: string;
  /** webrtc roomId carried for switchboard scoping parity. */
  originRoomId: string;
}

const DEFAULT_BOARDS: TsBoard[] = [
  {
    deviceId: '1cdbd49b9198',
    // RE-HOME 2026-10-18: old node 100.108.153.58 was removed from the
    // tailnet (control plane "node not found"); the board re-registered
    // with a fresh identity after the real auth key was baked into the
    // firmware. New VPN IP below (stable for this identity).
    vpnIp: process.env.TAILSCALE_ROUND_BOARD_IP ?? '100.81.53.32',
    sessionId: process.env.TS_VOICE_SESSION_ID ?? '669a3a68-f81e-4b40-b863-7a44b127a1ed',
    sessionKey:
      process.env.TS_VOICE_SESSION_KEY ?? 'agent:main:dashboard:38467751-6818-4c51-9099-6c954acd60db',
    originRoomId:
      '8509cfcd-0632-4bb1-a85e-93fb44f1132d:669a3a68-f81e-4b40-b863-7a44b127a1ed~1cdbd49b9198',
  },
  {
    // AMOLED (esp32-424cec): registered on the tailnet 2026-10-17 14:33,
    // LAN-direct with VM2. Voice WG lane uses the same :9100 protocol;
    // session = Room 2 (its NVS-restored default per the 14:26 re-home).
    deviceId: '7c0c5f424cec',
    vpnIp: process.env.TAILSCALE_AMOLED_BOARD_IP ?? '100.125.137.52',
    sessionId: process.env.TS_AMOLED_VOICE_SESSION_ID ?? '9a6a3d65-b127-476b-8010-b72037a4cb00',
    sessionKey:
      process.env.TS_AMOLED_VOICE_SESSION_KEY ?? 'agent:main:dashboard:c36f0e3b-ec09-469e-8b30-f609433d3b50',
    originRoomId:
      '8509cfcd-0632-4bb1-a85e-93fb44f1132d:9a6a3d65-b127-476b-8010-b72037a4cb00~7c0c5f424cec',
  },
];

interface BoardTurnState {
  stt: SttSessionLike | null;
  sttOpen: boolean;
  sttReady: boolean;
  sttAudioDone: boolean;  // arrived before infer session ready (remote-PTT diag path)
  turnBusy: boolean;
  turnEpoch: number;      // FIELD 2026-10-19 12:45: bumped by stt.cancel/reply.cancel —
                          // in-flight STT onDone/chat/TTS capture this and bail (a
                          // slide-cancel must also kill the turn the audio already fed)
  tts: ElevenLabsTtsSession | null;
  pendingPcm: Buffer[];   // arrived before the infer session was ready
  lastSeen: number;
}

export class TailscaleVoiceServer {
  private readonly sock = createSocket('udp4');
  private readonly boards = new Map<string, TsBoard>();
  private readonly turns = new Map<string, BoardTurnState>();
  /* FIELD 2026-10-19 13:20: boards wanting sessions.list pushes (session color
   * for the breathing/plankton) — re-pushed on the shared model's interval. */
  private readonly sessionsSubscribed = new Set<string>();
  private sessionsInterval: NodeJS.Timeout | null = null;

  constructor(boards: TsBoard[] = DEFAULT_BOARDS) {
    for (const b of boards) this.boards.set(b.vpnIp, b);
  }

  start(): void {
    this.sock.on('message', (msg, rinfo: RemoteInfo) => this.onPacket(msg, rinfo));
    this.sock.on('error', (err) => {
      console.error(`[ts-voice] socket error: ${err.message}`);
      try { this.sock.close(); } catch { /* ignore */ }
    });
    this.sock.bind(WG_VOICE_PORT, () => {
      console.log(`[ts-voice] WG voice lane listening on udp/${WG_VOICE_PORT} (${this.boards.size} board(s) allowlisted)`);
      // One-shot VISIBLE probe: the board's dispatcher logs unknown types at
      // INFO — a single serial line proves the VM:9100→board:9100 path per
      // daemon start. The periodic keepalive below stays SILENT (board
      // handles "keepalive" without logging): 6 lines/min would violate the
      // USB-CDC serial discipline. Establish the flow immediately either way.
      for (const [vpnIp, board] of this.boards) {
        this.sock.send(Buffer.from('{"t":"__flow_probe__"}'), WG_VOICE_PORT, vpnIp);
      }
      this.flowKeepalive();
      const timer = setInterval(() => this.flowKeepalive(), FLOW_KEEPALIVE_MS);
      timer.unref();
      /* FIELD 13:30: unconditional session-color feed starts with the lane
       * (no subscription state to lose across daemon restarts). */
      this.ensureSessionsInterval();
    });
    this.sock.unref();
  }

  /** Keep board:9100 → VM:9100 deliverable: tailscaled forwards inbound UDP
   * only for recently-sent flows. Payload: daemon.hello (not keepalive) —
   * the board's dispatcher flips its animation gate on it (field 16:3x:
   * the WG-primary lane never delivered hello so the gate stayed shut and
   * the boot display stuck; idempotent + sticky on the board). */
  private flowKeepalive(): void {
    for (const [vpnIp, board] of this.boards) {
      this.sock.send(Buffer.from('{"t":"daemon.hello"}'), WG_VOICE_PORT, vpnIp, (err) => {
        if (err) console.warn(`[ts-voice] hello/keepalive to ${board.deviceId}: ${err.message}`);
      });
    }
  }

  private onPacket(msg: Buffer, rinfo: RemoteInfo): void {
    const board = this.boards.get(rinfo.address);
    if (!board) {
      console.warn(`[ts-voice] dropped packet from non-allowlisted ${rinfo.address}`);
      return;
    }
    if (msg.length === 0) return;

    if (msg[0] === 0x7b /* '{' */) {
      let control: Record<string, unknown>;
      try {
        control = JSON.parse(msg.toString('utf8'));
      } catch {
        return;
      }
      this.onControl(board, control, rinfo);
    } else {
      this.onPcm(board, msg);
    }
  }

  private send(board: TsBoard, obj: Record<string, unknown>): void {
    const payload = Buffer.from(JSON.stringify(obj));
    this.sock.send(payload, WG_VOICE_PORT, board.vpnIp);
  }

  private sendPcm(board: TsBoard, pcm: Uint8Array): void {
    this.sock.send(Buffer.from(pcm), WG_VOICE_PORT, board.vpnIp);
  }

  private turn(board: TsBoard): BoardTurnState {
    let t = this.turns.get(board.vpnIp);
    if (!t) {
      t = { stt: null, sttOpen: false, sttReady: false, sttAudioDone: false, turnBusy: false, turnEpoch: 0, tts: null, pendingPcm: [], lastSeen: 0 };
      this.turns.set(board.vpnIp, t);
    }
    t.lastSeen = Date.now();
    return t;
  }

  private onControl(board: TsBoard, msg: Record<string, unknown>, rinfo: RemoteInfo): void {
    const t = this.turn(board);
    const type = msg.t as string;

    switch (type) {
      case 'stt.start': {
        if (t.stt) { try { t.stt.close(); } catch { /* ignore */ } t.stt = null; }
        console.log(`[ts-voice ${board.deviceId}] opening infer STT session (WG lane, ${rinfo.address})`);
        this.send(board, daemonToPhone.sttReady());
        // Reset turn state BEFORE constructing the session: OpenClawInfer-
        // SttSession fires onReady SYNCHRONOUSLY during construction (field
        // 13:14:28 — 'ready' logged the same ms as 'opening'), so resetting
        // after `new` would clobber sttReady=true and strand the whole turn
        // (PCM buffers forever, audio.done defers to an onReady that already
        // fired — the 13:14 deadlock).
        t.sttOpen = true;
        t.sttReady = false;
        t.sttAudioDone = false;
        t.pendingPcm = [];
        let sessionRef: SttSessionLike | null = null;
        try {
          const session = new OpenClawInferSttSession(
            { sampleRate: STT_SAMPLE_RATE },
            {
              onReady: () => {
                console.log(`[ts-voice ${board.deviceId}] infer STT ready — flushing ${t.pendingPcm.length} buffered chunk(s)`);
                t.sttReady = true;
                for (const b of t.pendingPcm.splice(0)) {
                  sessionRef?.sendAudio(b);
                }
                // audio-done raced the session-create (remote-PTT diag: the
                // board ends its window on its own schedule): flush, then
                // signal done — mirrors the rendezvous wrapper's ordering.
                if (t.sttAudioDone) {
                  console.log(`[ts-voice ${board.deviceId}] audio.done raced create — signaling done after flush`);
                  void Promise.resolve(sessionRef?.signalAudioDone());
                }
              },
              onPartial: (text, isFinal) => {
                this.send(board, { t: 'stt.partial', text, is_final: !!isFinal });
              },
              onDone: (text: string) => {
                console.log(`[ts-voice ${board.deviceId}] STT done: "${text.slice(0, 80)}"`);
                // FIELD 2026-10-19 12:45: epoch guard — a slide-cancel that
                // arrived while infer was spawning (close() on a spawning
                // session doesn't stop its pending onDone) must not respawn
                // the turn here.
                const epochAtDone = t.turnEpoch;
                this.send(board, daemonToPhone.sttDone(text));
                void this.runVoiceTurn(board, text, epochAtDone);
              },
              onError: (code: string) => {
                console.error(`[ts-voice ${board.deviceId}] STT error: ${code}`);
                this.send(board, { t: 'stt.error', message: code });
              },
              onClosed: () => {
                if (t.stt && sessionRef && t.stt === sessionRef) t.stt = null;
              },
            },
          );
          sessionRef = session;
          t.stt = session;
        } catch (err) {
          console.error(`[ts-voice ${board.deviceId}] STT create failed: ${err instanceof Error ? err.message : err}`);
          this.send(board, { t: 'stt.error', message: 'openclaw_infer_stt_failed' });
        }
        return;
      }

      case 'stt.audio.done': {
        if (t.stt) {
          if (t.sttReady) {
            void Promise.resolve(t.stt.signalAudioDone());
          } else {
            // Session still spawning — flag it; onReady flushes PCM then signals.
            t.sttAudioDone = true;
            console.log(`[ts-voice ${board.deviceId}] audio.done before ready — deferred`);
          }
        }
        return;
      }

      case 'stt.cancel':
      case 'reply.cancel': {
        // FIELD 2026-10-19 12:45 (slide-cancel semantics): a cancelled turn
        // must ALSO kill whatever the already-uplinked audio started. Bump
        // the epoch (in-flight onDone/speak bail below), then drop the
        // session + TTS. The turnBusy reset only matters for the case where
        // no chat was running yet.
        const epochAtCancel = ++t.turnEpoch;
        console.log(`[ts-voice ${board.deviceId}] stt.cancel — turn epoch=${epochAtCancel}, dropping session${t.turnBusy ? ' (chat in flight: it will bail on the epoch guard)' : ''}`);
        if (t.stt) { try { t.stt.close(); } catch { /* ignore */ } t.stt = null; }
        if (t.tts) { try { t.tts.cancel(); } catch { /* ignore */ } t.tts = null; }
        t.pendingPcm = [];
        t.sttOpen = false;
        t.sttReady = false;
        t.sttAudioDone = false;
        t.turnBusy = false;
        return;
      }

      case 'keepalive':
        return; // silence on the lane is fine; state is connectionless UDP

      case 'sessions.list.request':
        void this.sendRecentSessions(board);
        return;

      case 'sessions.list.subscribe': {
        this.sessionsSubscribed.add(board.vpnIp);
        this.ensureSessionsInterval();
        void this.sendRecentSessions(board);
        return;
      }

      case 'sessions.list.unsubscribe':
        this.sessionsSubscribed.delete(board.vpnIp);
        return;

      default:
        // sessions.catalog / destinations / create + anything else: the WG
        // lane serves list+color (this board's need); the rest stay
        // rendezvous-lane features.
        return;
    }
  }

  /* FIELD 2026-10-19 13:20: sessions.list serving — session color for the
   * breathing/plankton (never sent on the WG lane; board stayed on its
   * compiled-in fallback blue). Mirrors voiceSession.sendRecentSessions:
   * same provider, same 60s re-push interval on subscription. */
  private async sendRecentSessions(board: TsBoard): Promise<void> {
    try {
      const snapshot = await defaultRecentSessionsCache.get();
      const ours = snapshot.sessions.find(s => s.sessionId === board.sessionId);
      console.log(`[ts-voice ${board.deviceId}] sessions.list served: ${snapshot.sessions.length} session(s); ours=${board.sessionId.slice(0, 8)} color=${ours?.color ?? '(none set)'}`);
      this.send(board, daemonToPhone.sessionsList(snapshot));
    } catch {
      console.error(`[ts-voice ${board.deviceId}] sessions.list provider failed — empty snapshot sent`);
      this.send(board, daemonToPhone.sessionsList(createEmptyRecentSessionsSnapshot()));
    }
  }

  private ensureSessionsInterval(): void {
    if (this.sessionsInterval) return;
    this.sessionsInterval = setInterval(() => {
      /* FIELD 2026-10-19 13:30 (restart race: the board subscribed once on
       * attach, but a daemon restart wipes ANY server-side subscription
       * state — the color feed died with the dead daemon). Serve
       * UNCONDITIONALLY to every allowlisted board: the daemon knows its
       * boards at bind time, so restarts self-heal within one interval. */
      for (const board of this.boards.values()) {
        void this.sendRecentSessions(board);
      }
    }, 60_000);   /* mirrors voiceSession RECENT_SESSIONS_SUBSCRIPTION_INTERVAL_MS */
    this.sessionsInterval.unref?.();
  }

  private onPcm(board: TsBoard, pcm: Buffer): void {
    const t = this.turn(board);
    if (t.stt && t.sttOpen && t.sttReady) {
      t.stt.sendAudio(new Uint8Array(pcm));
    } else if (t.stt && t.sttOpen) {
      // Infer session spawning (CLI takes seconds) — buffer like the
      // rendezvous lane's pendingAudio pattern; flushed on onReady.
      if (t.pendingPcm.length < 400) t.pendingPcm.push(Buffer.from(pcm));
    }
  }

  /** The chat half: identical shape to voiceSession.runReplyTurn, WG transport. */
  private async runVoiceTurn(board: TsBoard, userText: string, epochAtCall?: number): Promise<void> {
    const t = this.turn(board);
    // FIELD 2026-10-19 12:45: a slide-cancel between STT completion and chat
    // start bumps the epoch — this call is for a dead turn, drop it.
    if (epochAtCall !== undefined && epochAtCall !== t.turnEpoch) {
      console.log(`[ts-voice ${board.deviceId}] chat dropped (cancel raced: epoch ${epochAtCall} != ${t.turnEpoch}) — "${userText.slice(0, 40)}"`);
      return;
    }
    if (t.turnBusy) {
      console.warn(`[ts-voice ${board.deviceId}] turn already running — dropped "${userText.slice(0, 40)}"`);
      return;
    }
    if (!userText.trim()) {
      // Ambient-room recording (remote-PTT diag) can legitimately transcribe
      // to nothing — that IS a successful uplink test, just a quiet room.
      console.log(`[ts-voice ${board.deviceId}] empty transcript (ambient quiet) — uplink validated, no chat turn`);
      return;
    }
    t.turnBusy = true;
    const started = Date.now();
    try {
      const result = await runChat(userText, {
        sessionId: board.sessionId,
        originRoomId: board.originRoomId,
        sessionKey: board.sessionKey,
      });
      // FIELD 2026-10-19 12:45: chat finished AFTER a slide-cancel — the
      // reply belongs to a cancelled turn; do not open TTS.
      if (epochAtCall !== undefined && epochAtCall !== t.turnEpoch) {
        console.log(`[ts-voice ${board.deviceId}] chat done but cancelled — epoch ${epochAtCall} != ${t.turnEpoch}; reply discarded`);
        return;
      }
      const replyText = result.text;
      console.log(`[ts-voice ${board.deviceId}] chat done (${Date.now() - started}ms): "${replyText.slice(0, 80)}" opening TTS`);
      this.send(board, daemonToPhone.replyDone(replyText));
      await this.speak(board, replyText);
    } catch (err) {
      const code = err instanceof Error && 'code' in err ? String((err as { code?: string }).code) : 'reply_failed';
      console.error(`[ts-voice ${board.deviceId}] reply failed: ${code}`);
      this.send(board, daemonToPhone.replyError(code));
    } finally {
      t.turnBusy = false;
    }
  }

  private speak(board: TsBoard, text: string): Promise<void> {
    const t = this.turn(board);
    const ttsStart = Date.now();
    let firstSendAt = 0;      // when the first PCM chunk left the daemon
    let firstPcmLogged = false;
    let synthDone = false;
    let pending = Buffer.alloc(0);
    let sentChunks = 0;
    let drainTimer: NodeJS.Timeout | null = null;
    let doneTimer: NodeJS.Timeout | null = null;
    let settled = false;

    return new Promise((resolve) => {
      this.send(board, daemonToPhone.ttsStart(16000, { text: text.slice(0, 80) }));
      const finish = () => {
        if (settled) return;
        settled = true;
        if (drainTimer) { clearInterval(drainTimer); drainTimer = null; }
        if (doneTimer) { clearTimeout(doneTimer); doneTimer = null; }
        t.tts = null;
        const audioMs = sentChunks * PCM_CHUNK_MS;
        console.log(`[ts-voice ${board.deviceId}] TTS done (total ${Date.now() - ttsStart}ms, chunks=${sentChunks} ≈${Math.round(audioMs)}ms audio, tts.done sent at board play-out)`);
        this.send(board, daemonToPhone.ttsDone());
        // Double-send tts.done (300ms apart): idempotent on the board
        // (handler no-ops on a drained pipeline), belt-and-braces against a
        // control packet lost at the burst tail.
        setTimeout(() => {
          try { this.send(board, daemonToPhone.ttsDone()); } catch { /* ignore */ }
        }, 300).unref?.();
        resolve();
      };
      // TTS.DONE DEFERRAL (field 14:08): the board's amp schedule cuts audio
      // at tts.done — if the daemon finishes DELIVERING before the board
      // finishes PLAYING, the buffered tail dies silently (14:08: 13.1s
      // audio delivered in 7.9s wall, board played 8.4s, ~4.7s cut; the
      // 13:46 count test lost numbers the same way). The board's ring
      // absorbs delivery overshoot (16s PSRAM), so the correct contract:
      // send tts.done only after first-send + audio-duration — the moment
      // the board has played out everything it received. +400ms margin.
      const scheduleBoardPlayOut = () => {
        if (doneTimer || settled) return;
        // Gate on FULLY drained: synth done AND every chunk sent. Field bug
        // 14:14 (self-caught): scheduling on the first post-synth tick used
        // the partial chunk count (231 of 449) and fired tts.done while 8.2s
        // of audio was still in flight — the exact tail-cut this feature
        // exists to prevent.
        if (!synthDone || pending.length > 0) return;
        if (!firstSendAt) { finish(); return; }  // zero-length audio edge
        const audioMs = sentChunks * PCM_CHUNK_MS;
        const playedOutAt = firstSendAt + audioMs + 400;
        const waitMs = Math.max(0, playedOutAt - Date.now());
        if (waitMs > 0) {
          console.log(`[ts-voice ${board.deviceId}] delivery complete (${sentChunks} chunks ≈${Math.round(audioMs)}ms audio) — tts.done deferred ${Math.round(waitMs)}ms for board play-out`);
        }
        doneTimer = setTimeout(finish, waitMs);
        doneTimer.unref?.();
      };
      // PACED EMISSION (field 13:46, Fredrik count test): the reply text was
      // complete but ~1-1.5s of audio went missing with NO pauses — ElevenLabs
      // flash synthesizes FASTER than realtime (2.7x measured: 15.7s audio in
      // 5.85s wall), and wire-speed drain burst-dropped chunks at the board's
      // RX queue. MUST BE ~1x REALTIME: the board's amp drain-grace is ~3s,
      // so if arrival outruns playback the buffered tail dies when tts.done
      // cuts the amp (13:59: 6.3s played of 15.7s delivered — the board's
      // park cycles were the symptom, the wall-clock overshoot the cause).
      // 37ms tick, 1 chunk (1200B = 37.5ms audio): delivery ≈ playback rate,
      // tts.done lands as the tail finishes, amp grace covers ~nothing.
      const drain = () => {
        const budget = pending.length > PCM_CHUNK_BYTES * 8 ? 2 : 1;
        let b = budget;
        while (b > 0 && pending.length > 0) {
          const n = Math.min(PCM_CHUNK_BYTES, pending.length);
          if (!firstSendAt) {
            firstSendAt = Date.now();
            console.log(`[ts-voice ${board.deviceId}] first TTS PCM sent (+${firstSendAt - ttsStart}ms)`);
          }
          firstPcmLogged = true;
          this.sendPcm(board, pending.subarray(0, n));
          pending = pending.subarray(n);
          sentChunks += 1;
          b -= 1;
        }
        if (synthDone) scheduleBoardPlayOut();
      };
      drainTimer = setInterval(drain, 37);
      drainTimer.unref?.();
      void drain();
      const session = new ElevenLabsTtsSession(
        { text },
        {
          onAudio: (pcm) => {
            if (settled) return;
            pending = Buffer.concat([pending, Buffer.from(pcm)]);
          },
          onDone: () => {
            if (settled) return;
            synthDone = true;
            void drain();
          },
          onError: (message: string) => {
            console.error(`[ts-voice ${board.deviceId}] TTS error: ${message}`);
            this.send(board, { t: 'tts.error', message });
            finish();
          },
        },
      );
      t.tts = session;
    });
  }
}

let started = false;
export function startTailscaleVoiceServer(): void {
  if (started) return;
  started = true;
  // Minute-gated with the monitor: the lane itself is cheap; only boards in
  // DEFAULT_BOARDS are accepted (identity pinned at the source).
  if (process.env.TS_VOICE_SERVER_DISABLE === '1') {
    console.log('[ts-voice] disabled via TS_VOICE_SERVER_DISABLE');
    return;
  }
  new TailscaleVoiceServer().start();
}