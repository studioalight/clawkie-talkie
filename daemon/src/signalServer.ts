// The hosted rambly-style signaling broker the daemon talks to. Not user
// configurable — runtime always uses this default. Internal callers may
// still pass a different value when wiring tests/internal harnesses.
// Default signaling broker. Overridden everywhere by CT_SIGNAL_SERVER /
// --signal-server (all real deployments set it); this fallback exists so a
// bare dev run still works. Never default to third-party hosted infra —
// point at our own broker (2026-10-07: api.rambly.app default removed).
export const DEFAULT_SIGNAL_SERVER = 'https://signaling.studioalight.com';
