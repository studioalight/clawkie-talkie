/**
 * tailscaleTransport — direct daemon↔board transport over the tailnet.
 *
 * The rendezvous/DERP path exists because boards lived behind NAT with only
 * a public signaling host. With microlink (spike 2026-10-17: esp32-9b9198 on
 * 100.108.153.58, LAN-direct, med echo RTT 22–39ms at sustainable rates) the
 * board has a stable always-dialable address — this module is the daemon's
 * seam for that second architecture: no signaling server, no zombie-lane
 * class of failure. Voice transport itself waits for the firmware
 * integration (microlink component in clawkie fw); TODAY this module proves
 * and monitors the lane: HTTP status/health + UDP echo RTT, both dialed
 * directly at the tailnet IP.
 *
 * Rate discipline: the SPIKE example echoes at ~23 replies/s (its simple
 * UDP app path — 2026-10-17 field ceiling, NOT the WG plane; the WG plane
 * delivered 75/s with p90 under 100ms). The monitor therefore runs at
 * minute-cadence; explicit bursts stay low-rate until real firmware lifts
 * the ceiling.
 *
 * Board identity: deviceId (factory MAC lane, e.g. 1cdbd49b9198) → tailnet
 * endpoint. The board's own /api/status is authoritative for its vpn_ip —
 * registerDevice() can resolve it at startup and re-verify if the monitor
 * goes dark (Tailscale re-assigns tailscale IPs only on re-auth; the IP is
 * stable across reboots for the same node identity).
 */

import { createSocket } from 'node:dgram';
import { get } from 'node:http';

export interface TailnetBoardConfig {
  /** Factory-MAC device lane id the daemon already knows boards by. */
  deviceId: string;
  /** Tailnet node identity label ( MagicDNS short name, e.g. esp32-9b9198). */
  node?: string;
  /** Tailnet IP. Optional if node is set (resolved once via status API). */
  vpnIp?: string;
  /** UDP echo port (spike example: 9000). */
  echoPort?: number;
  /** HTTP port (spike example: 80). */
  httpPort?: number;
}

export interface BoardStatus {
  vpnIp: string | null;
  state: string | null;
  peerCount: number;
  freeHeap: number | null;
  freePsram: number | null;
  connection: string | null;
  raw: unknown;
}

export type BoardLaneState = 'unknown' | 'direct' | 'degraded' | 'lost';

export interface LaneHealth {
  state: BoardLaneState;
  echoRttMs: number | null;
  lossPct: number | null;
  heapFree: number | null;
  checkedAt: number;
  note?: string;
}

const HTTP_TIMEOUT_MS = 4000;
const ECHO_TIMEOUT_MS = 2500;
const MONITOR_INTERVAL_MS = 60_000;

export class TailscaleTransport {
  private readonly boards = new Map<string, TailnetBoardConfig & { vpnIp: string }>();
  private monitorTimer: NodeJS.Timeout | null = null;

  registerDevice(cfg: TailnetBoardConfig): void {
    if (!cfg.vpnIp && !cfg.node) {
      throw new Error('tailscaleTransport: need vpnIp or node to dial');
    }
    const vpnIp = cfg.vpnIp ?? cfg.deviceId; // placeholder; resolved at first probe
    this.boards.set(cfg.deviceId, { ...cfg, vpnIp });
  }

  /** Resolve the board's authoritative vpn_ip via its status API (when node-only). */
  async resolveVpnIp(cfgName: string): Promise<string | null> {
    const cfg = this.boards.get(cfgName);
    if (!cfg?.node) return cfg?.vpnIp ?? null;
    // MagicDNS name is dialable directly on a tailnet host; the status API
    // returns the canonical vpn_ip.
    const host = `${cfg.node}`;
    try {
      const url = `http://${host}/api/status`;
      const body = await this.httpGetJson(url);
      const vpnIp = (body as { vpn_ip?: string })?.vpn_ip ?? null;
      if (vpnIp && this.boards.has(cfgName)) this.boards.get(cfgName)!.vpnIp = vpnIp;
      return vpnIp;
    } catch {
      return null;
    }
  }

  /** Authoritative board state over HTTP (the control lane). */
  async status(deviceId: string): Promise<BoardStatus | null> {
    const cfg = this.boards.get(deviceId);
    if (!cfg) return null;
    try {
      const body = (await this.httpGetJson(`http://${cfg.vpnIp}/api/status`)) as Record<string, unknown>;
      const st: BoardStatus = {
        vpnIp: (body.vpn_ip as string) ?? cfg.vpnIp,
        state: (body.state as string) ?? null,
        peerCount: (body.peer_count as number) ?? 0,
        freeHeap: (body.free_heap as number) ?? null,
        freePsram: (body.free_psram as number) ?? null,
        connection: (body.connection as string) ?? null,
        raw: body,
      };
      // keep the working address (vpn_ip may have been resolved by board itself)
      if (st.vpnIp) cfg.vpnIp = st.vpnIp;
      return st;
    } catch {
      return null;
    }
  }

  /** One timestamped UDP echo probe (the data-plane lane, real WG path). */
  async echoOnce(deviceId: string): Promise<number | null> {
    const cfg = this.boards.get(deviceId);
    if (!cfg) return null;
    const port = (cfg.echoPort ?? 9000);
    const sock = createSocket('udp4');
    const t0 = Date.now();
    const payload = Buffer.from(`TSPOKE${t0}`);
    return await new Promise((resolve) => {
      const finish = (rtt: number | null) => {
        try { sock.close(); } catch { /* ignore */ }
        resolve(rtt);
      };
      const timer = setTimeout(() => finish(null), ECHO_TIMEOUT_MS);
      sock.on('message', (msg) => {
        clearTimeout(timer);
        if (!msg.toString('utf8').startsWith('ECHO:')) { finish(null); return; }
        finish(Date.now() - t0);
      });
      sock.on('error', () => { clearTimeout(timer); finish(null); });
      sock.send(payload, port, cfg.vpnIp, (err) => {
        if (err) { clearTimeout(timer); finish(null); }
      });
    });
  }

  /** Quick lane health: status + one echo. Minute-cadence from the monitor. */
  async health(deviceId: string): Promise<LaneHealth> {
    const st = await this.status(deviceId);
    const rtt = await this.echoOnce(deviceId);
    const state: BoardLaneState =
      st === null ? 'lost' : st.state === 'CONNECTED' ? 'direct' : 'degraded';
    return {
      state,
      echoRttMs: rtt,
      lossPct: rtt === null ? 100 : 0,
      heapFree: st?.freeHeap ?? null,
      checkedAt: Date.now(),
      note: st?.connection ?? undefined,
    };
  }

  /** Minute-cadence monitor; emits health snapshots. Unref'd — never holds the process. */
  startMonitor(onHealth: (deviceId: string, health: LaneHealth) => void): void {
    if (this.monitorTimer) return;
    const tick = () => {
      for (const [deviceId] of this.boards) {
        void this.health(deviceId).then(
          (h) => {
            onHealth(deviceId, h);
            // Address may have been corrected by the status probe.
            const cfg = this.boards.get(deviceId);
            console.log(
              `[tailnet ${deviceId}] ${h.state} echo=${h.echoRttMs ?? '—'}ms heap=${h.heapFree ?? '—'} at ${cfg?.vpnIp}`,
            );
          },
          () => {
            onHealth(deviceId, { state: 'lost', echoRttMs: null, lossPct: 100, heapFree: null, checkedAt: Date.now() });
          },
        );
      }
    };
    tick(); // immediate first reading
    this.monitorTimer = setInterval(tick, MONITOR_INTERVAL_MS);
    this.monitorTimer.unref?.();
  }

  stopMonitor(): void {
    if (this.monitorTimer) { clearInterval(this.monitorTimer); this.monitorTimer = null; }
  }

  private httpGetJson(url: string): Promise<Record<string, unknown>> {
    return new Promise((resolve, reject) => {
      const req = get(url, { timeout: HTTP_TIMEOUT_MS }, (res) => {
        let data = '';
        res.on('data', (c) => { data += c; });
        res.on('end', () => {
          try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
        });
      });
      req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
      req.on('error', reject);
    });
  }
}

/** Live tailnet nodes for clawkie boards (spike-era registry; firmware
 * integration replaces echoPort with the voice port once it exists). */
export function defaultBoardRegistry(): TailscaleTransport {
  const t = new TailscaleTransport();
  // Round board in spike state (2026-10-17): esp32-9b9198 @ 100.108.153.58
  t.registerDevice({
    deviceId: '1cdbd49b9198',
    node: 'esp32-9b9198',
    vpnIp: process.env.TAILSCALE_ROUND_BOARD_IP ?? '100.108.153.58',
    echoPort: 9000,
    httpPort: 80,
  });
  return t;
}