/**
 * wrtc-patch.ts — node-datachannel polyfill with simple-peer compatibility
 */

import ndc from 'node-datachannel';
import { default as ndcPolyfill } from 'node-datachannel/polyfill';

// --- Patch RTCSessionDescription.sdp to be writable ---

const RTCSessionDescriptionCtor = (ndcPolyfill as any).RTCSessionDescription;
if (RTCSessionDescriptionCtor?.prototype) {
  const SDP_PROTO = RTCSessionDescriptionCtor.prototype;
  const sdpDescriptor = Object.getOwnPropertyDescriptor(SDP_PROTO, 'sdp');
  if (sdpDescriptor?.get && !sdpDescriptor.set) {
    const origGetter = sdpDescriptor.get;
    Object.defineProperty(SDP_PROTO, 'sdp', {
      get() { return origGetter.call(this); },
      set(val: string) {
        Object.defineProperty(this, 'sdp', {
          get() { return val; },
          configurable: true,
          enumerable: true,
        });
      },
      configurable: true,
      enumerable: true,
    });
  }
}

// --- Patched RTCPeerConnection that reliably gathers TURN relay candidates ---

const OrigRTCPeerConnection = (ndcPolyfill as any).RTCPeerConnection;

export class PatchedRTCPeerConnection extends OrigRTCPeerConnection {
  private _nativePC: any = null;

  constructor(config: any = {}) {
    console.log('[wrtc-patch] PatchedRTCPeerConnection instantiated! config.iceServers count:', config.iceServers?.length);
    let nativePC = config.peerConnection;
    if (!nativePC) {
      const iceServers = config.iceServers?.map((s: any) => {
        const urls = Array.isArray(s.urls) ? s.urls : [s.urls];
        return urls.map((u: string) => {
          if (s.username && s.credential) {
            const [proto, rest] = u.split(/:(.*)/);
            return `${proto}:${s.username}:${s.credential}@${rest}`;
          }
          return u;
        });
      }).flat() ?? [];

      console.log('[wrtc-patch] creating native ndc.PeerConnection with:', JSON.stringify(iceServers));

      const peerIdentity = config.peerIdentity ?? `peer-${Math.random().toString(36).slice(2, 9)}`;
      const nativeConfig: any = { iceServers };
      if (config.iceTransportPolicy) {
        nativeConfig.iceTransportPolicy = config.iceTransportPolicy;
      }

      nativePC = new (ndc as any).PeerConnection(peerIdentity, nativeConfig);
      super({ ...config, peerConnection: nativePC });
    } else {
      super(config);
    }
    this._nativePC = nativePC;
  }

  async setRemoteDescription(description: any): Promise<void> {
    await super.setRemoteDescription(description);

    // If remote SDP contains inline ICE candidates (typical for embedded esp_peer answers),
    // extract and feed them directly to native libdatachannel so it adds them to the ICE checklist.
    if (this._nativePC && typeof description?.sdp === 'string') {
      const lines = description.sdp.split(/\r?\n/);
      for (const line of lines) {
        if (line.startsWith('a=candidate:')) {
          const candStr = line.slice(2); // strip leading "a="
          try {
            console.log('[wrtc-patch] Feeding inline candidate to nativePC:', candStr);
            this._nativePC.addRemoteCandidate(candStr, '0');
          } catch (err) {
            console.warn('[wrtc-patch] Failed to feed inline candidate:', err);
          }
        }
      }
    }
  }
}

export const wrtc = {
  ...ndcPolyfill,
  RTCPeerConnection: PatchedRTCPeerConnection,
} as unknown as import('simple-peer').Options['wrtc'];

export default wrtc;
