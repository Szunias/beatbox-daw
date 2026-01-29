/**
 * Transport Store
 * Zustand store for managing DAW transport (playback) state
 * Syncs with backend via WebSocket
 */

import { create } from 'zustand';
import { LoopRegion, TransportState, TICKS_PER_BEAT } from '../types/project';

// WebSocket send function type
type WebSocketSendFn = (type: string, data?: Record<string, unknown>) => void;

interface TransportStoreState {
  // Transport state
  state: TransportState;
  currentTick: number;
  startTime: number | null;

  // Loop
  loopRegion: LoopRegion;

  // Metronome
  clickEnabled: boolean;
  clickVolume: number;

  // Recording
  recordStartTick: number | null;
  preRollEnabled: boolean;
  preRollBars: number;

  // Playback
  isFollowPlayhead: boolean;

  // WebSocket connection
  _wsSend: WebSocketSendFn | null;
  _isConnected: boolean;
}

interface TransportActions {
  // Transport controls
  play: () => void;
  pause: () => void;
  stop: () => void;
  record: () => void;

  // Position
  setCurrentTick: (tick: number) => void;
  seekTo: (tick: number) => void;
  seekToBar: (bar: number, numerator: number) => void;
  jumpForward: (ticks: number) => void;
  jumpBackward: (ticks: number) => void;

  // Loop
  setLoopEnabled: (enabled: boolean) => void;
  setLoopRegion: (startTick: number, endTick: number) => void;

  // Metronome
  toggleClick: () => void;
  setClickVolume: (volume: number) => void;

  // Recording
  setPreRoll: (enabled: boolean, bars?: number) => void;

  // Playback
  toggleFollowPlayhead: () => void;

  // Internal
  updatePlaybackPosition: (tick: number) => void;

  // WebSocket sync
  setWebSocket: (sendFn: WebSocketSendFn | null, isConnected: boolean) => void;
  syncFromBackend: (tick: number, state: TransportState) => void;
}

export const useTransportStore = create<TransportStoreState & TransportActions>()((set, get) => ({
  // Initial state
  state: 'stopped',
  currentTick: 0,
  startTime: null,

  loopRegion: {
    enabled: false,
    startTick: 0,
    endTick: TICKS_PER_BEAT * 16, // 4 bars at 4/4
  },

  clickEnabled: true,
  clickVolume: 0.7,

  recordStartTick: null,
  preRollEnabled: false,
  preRollBars: 1,

  isFollowPlayhead: true,

  _wsSend: null,
  _isConnected: false,

  // === Transport Controls ===
  play: () => {
    const { state: currentState, _wsSend, _isConnected } = get();
    if (currentState === 'playing') return;

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('transport_play');
    }

    // Update local state immediately for responsive UI
    set({
      state: 'playing',
      startTime: Date.now(),
    });
  },

  pause: () => {
    const { state: currentState, _wsSend, _isConnected } = get();
    if (currentState !== 'playing' && currentState !== 'recording') return;

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('transport_pause');
    }

    set({
      state: 'paused',
      startTime: null,
    });
  },

  stop: () => {
    const { _wsSend, _isConnected } = get();

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('transport_stop');
    }

    set({
      state: 'stopped',
      currentTick: 0,
      startTime: null,
      recordStartTick: null,
    });
  },

  record: () => {
    const { state: currentState, _wsSend, _isConnected, currentTick } = get();

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('transport_record');
    }

    if (currentState === 'recording') {
      // Stop recording
      set({
        state: 'stopped',
        startTime: null,
        recordStartTick: null,
      });
    } else {
      // Start recording
      set({
        state: 'recording',
        startTime: Date.now(),
        recordStartTick: currentTick,
      });
    }
  },

  // === Position ===
  setCurrentTick: (tick) => {
    set({ currentTick: Math.max(0, tick) });
  },

  seekTo: (tick) => {
    const { _wsSend, _isConnected } = get();
    const newTick = Math.max(0, tick);

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('transport_seek', { tick: newTick });
    }

    set({ currentTick: newTick });
  },

  seekToBar: (bar, numerator) => {
    const tick = bar * TICKS_PER_BEAT * numerator;
    set({ currentTick: Math.max(0, tick) });
  },

  jumpForward: (ticks) => {
    set((state) => ({ currentTick: Math.max(0, state.currentTick + ticks) }));
  },

  jumpBackward: (ticks) => {
    set((state) => ({ currentTick: Math.max(0, state.currentTick - ticks) }));
  },

  // === Loop ===
  setLoopEnabled: (enabled) => {
    const { _wsSend, _isConnected, loopRegion } = get();

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('set_loop', {
        enabled,
        start_tick: loopRegion.startTick,
        end_tick: loopRegion.endTick
      });
    }

    set((state) => ({
      loopRegion: { ...state.loopRegion, enabled },
    }));
  },

  setLoopRegion: (startTick, endTick) => {
    if (startTick >= endTick) return;
    const { _wsSend, _isConnected, loopRegion } = get();

    const newStartTick = Math.max(0, startTick);
    const newEndTick = Math.max(startTick + TICKS_PER_BEAT, endTick);

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('set_loop', {
        enabled: loopRegion.enabled,
        start_tick: newStartTick,
        end_tick: newEndTick
      });
    }

    set((state) => ({
      loopRegion: {
        ...state.loopRegion,
        startTick: newStartTick,
        endTick: newEndTick,
      },
    }));
  },

  // === Metronome ===
  toggleClick: () => {
    const { _wsSend, _isConnected, clickEnabled } = get();
    const newEnabled = !clickEnabled;

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('set_click', { enabled: newEnabled });
    }

    set({ clickEnabled: newEnabled });
  },

  setClickVolume: (volume) => {
    set({ clickVolume: Math.max(0, Math.min(1, volume)) });
  },

  // === Recording ===
  setPreRoll: (enabled, bars = 1) => {
    set({
      preRollEnabled: enabled,
      preRollBars: Math.max(1, Math.min(4, bars)),
    });
  },

  // === Playback ===
  toggleFollowPlayhead: () => {
    set((state) => ({ isFollowPlayhead: !state.isFollowPlayhead }));
  },

  // === Internal ===
  updatePlaybackPosition: (tick) => {
    const { loopRegion } = get();

    // Handle loop
    if (loopRegion.enabled && tick >= loopRegion.endTick) {
      set({ currentTick: loopRegion.startTick });
    } else {
      set({ currentTick: tick });
    }
  },

  // === WebSocket Sync ===
  setWebSocket: (sendFn, isConnected) => {
    set({
      _wsSend: sendFn,
      _isConnected: isConnected,
    });
  },

  syncFromBackend: (tick, state) => {
    // Update local state from backend without triggering WebSocket messages
    set({
      currentTick: tick,
      state: state,
      startTime: state === 'playing' || state === 'recording' ? Date.now() : null,
    });
  },
}));
