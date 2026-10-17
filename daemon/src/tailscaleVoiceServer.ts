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

const WG_VOICE_PORT = 9100;
// Tailscale tunnel MTU is 1280: UDP payload must be ≤1252B to avoid
// fragmentation loss (field-probed 2026-10-17: 1000B ✓, 1280B ✗).
// 1200B = 37.5ms @16k mono16, 28B headroom.
const PCM_CHUNK_BYTES = 1200;
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
    vpnIp: process.env.TAILSCALE_ROUND_BOARD_IP ?? '100.108.153.58',
    sessionId: process.env.TS_VOICE_SESSION_ID ?? '669a3a68-f81e-4b40-b863-7a44b127a1ed',
    sessionKey:
      process.env.TS_VOICE_SESSION_KEY ?? 'agent:main:dashboard:38467751-6818-4c51-9099-6c954acd60db',
    originRoomId:
      '8509cfcd-0632-4bb1-a85e-93fb44f1132d:669a3a68-f81e-4b40-b863-7a44b127a1ed~1cdbd49b9198',
  },
];

interface BoardTurnState {
  stt: SttSessionLike | null;
  sttOpen: boolean;
  sttReady: boolean;
  turnBusy: boolean;
  tts: ElevenLabsTtsSession | null;
  pendingPcm: Buffer[];   // arrived before the infer session was ready
  lastSeen: number;
}

export class TailscaleVoiceServer {
  private readonly sock = createSocket('udp4');
  private readonly boards = new Map<string, TsBoard>();
  private readonly turns = new Map<string, BoardTurnState>();

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
    });
    this.sock.unref();
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
      t = { stt: null, sttOpen: false, sttReady: false, turnBusy: false, tts: null, pendingPcm: [], lastSeen: 0 };
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
              },
              onPartial: (text, isFinal) => {
                this.send(board, { t: 'stt.partial', text, is_final: !!isFinal });
              },
              onDone: (text: string) => {
                console.log(`[ts-voice ${board.deviceId}] STT done: "${text.slice(0, 80)}"`);
                this.send(board, daemonToPhone.sttDone(text));
                void this.runVoiceTurn(board, text);
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
          t.sttOpen = true;
          t.sttReady = false;
        } catch (err) {
          console.error(`[ts-voice ${board.deviceId}] STT create failed: ${err instanceof Error ? err.message : err}`);
          this.send(board, { t: 'stt.error', message: 'openclaw_infer_stt_failed' });
        }
        return;
      }

      case 'stt.audio.done': {
        if (t.stt) {
          void Promise.resolve(t.stt.signalAudioDone());
        }
        return;
      }

      case 'stt.cancel':
      case 'reply.cancel': {
        if (t.stt) { try { t.stt.close(); } catch { /* ignore */ } t.stt = null; }
        if (t.tts) { try { t.tts.cancel(); } catch { /* ignore */ } t.tts = null; }
        t.turnBusy = false;
        return;
      }

      case 'keepalive':
        return; // silence on the lane is fine; state is connectionless UDP

      default:
        // sessions.list.subscribe and friends: acknowledge minimally, lane is
        // stateless — the board shows its session from its own persisted state.
        return;
    }
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
  private async runVoiceTurn(board: TsBoard, userText: string): Promise<void> {
    const t = this.turn(board);
    if (t.turnBusy) {
      console.warn(`[ts-voice ${board.deviceId}] turn already running — dropped "${userText.slice(0, 40)}"`);
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
    return new Promise((resolve) => {
      this.send(board, daemonToPhone.ttsStart(16000, { text: text.slice(0, 80) }));
      let settled = false;
      let pending = Buffer.alloc(0);
      const finish = () => {
        if (settled) return;
        settled = true;
        t.tts = null;
        this.send(board, daemonToPhone.ttsDone());
        resolve();
      };
      const session = new ElevenLabsTtsSession(
        { text },
        {
          onAudio: (pcm) => {
            if (settled) return;
            pending = Buffer.concat([pending, Buffer.from(pcm)]);
            while (pending.length >= PCM_CHUNK_BYTES) {
              this.sendPcm(board, pending.subarray(0, PCM_CHUNK_BYTES));
              pending = pending.subarray(PCM_CHUNK_BYTES);
            }
          },
          onDone: () => {
            if (pending.length > 0) this.sendPcm(board, pending);
            finish();
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