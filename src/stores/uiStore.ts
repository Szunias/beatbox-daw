/**
 * UI Store
 * Zustand store for managing UI state (viewport, zoom, panels, etc.)
 */

import { create } from 'zustand';
import { TimelineViewport, SnapSettings, SnapValue, TICKS_PER_BEAT } from '../types/project';

export type Tool = 'select' | 'draw' | 'erase' | 'slice' | 'mute';
export type Panel = 'timeline' | 'pianoroll' | 'mixer' | 'beatbox';

interface UIStoreState {
  // Viewport
  timelineViewport: TimelineViewport;

  // Snap
  snapSettings: SnapSettings;

  // Tools
  currentTool: Tool;

  // Panels
  activePanel: Panel;
  isPianoRollOpen: boolean;
  isMixerOpen: boolean;
  isBeatboxPanelOpen: boolean;

  // Selection
  selectedNoteIds: string[];

  // Piano Roll
  pianoRollClipId: string | null;
  pianoRollTrackId: string | null;
  pianoRollViewport: {
    startTick: number;
    endTick: number;
    topPitch: number;
    bottomPitch: number;
    zoom: number;
  };

  // Misc
  isPlaying: boolean; // Cached from transport for UI updates
}

interface UIActions {
  // Viewport
  setTimelineViewport: (viewport: Partial<TimelineViewport>) => void;
  zoomTimeline: (factor: number, centerTick?: number) => void;
  scrollTimeline: (deltaTick: number) => void;
  scrollTimelineVertical: (deltaPixels: number) => void;

  // Snap
  setSnapEnabled: (enabled: boolean) => void;
  setSnapValue: (value: SnapValue) => void;

  // Tools
  setTool: (tool: Tool) => void;

  // Panels
  setActivePanel: (panel: Panel) => void;
  togglePianoRoll: () => void;
  toggleMixer: () => void;
  toggleBeatboxPanel: () => void;
  openPianoRoll: (trackId: string, clipId: string) => void;
  closePianoRoll: () => void;

  // Selection
  selectNotes: (noteIds: string[]) => void;
  addToNoteSelection: (noteIds: string[]) => void;
  clearNoteSelection: () => void;

  // Piano Roll viewport
  setPianoRollViewport: (viewport: Partial<UIStoreState['pianoRollViewport']>) => void;
  zoomPianoRoll: (factor: number) => void;
  scrollPianoRoll: (deltaTick: number) => void;
  scrollPianoRollVertical: (deltaPitch: number) => void;

  // Misc
  setIsPlaying: (isPlaying: boolean) => void;
}

const DEFAULT_TIMELINE_VIEWPORT: TimelineViewport = {
  startTick: 0,
  endTick: TICKS_PER_BEAT * 32, // 8 bars at 4/4
  zoom: 0.1, // pixels per tick
  verticalScroll: 0,
};

const DEFAULT_PIANO_ROLL_VIEWPORT = {
  startTick: 0,
  endTick: TICKS_PER_BEAT * 4, // 1 bar
  topPitch: 84, // C6
  bottomPitch: 36, // C2
  zoom: 0.2,
};

export const useUIStore = create<UIStoreState & UIActions>()((set, get) => ({
  // Initial state
  timelineViewport: DEFAULT_TIMELINE_VIEWPORT,

  snapSettings: {
    enabled: true,
    value: '1/8',
  },

  currentTool: 'select',

  activePanel: 'timeline',
  isPianoRollOpen: false,
  isMixerOpen: true,
  isBeatboxPanelOpen: true,

  selectedNoteIds: [],

  pianoRollClipId: null,
  pianoRollTrackId: null,
  pianoRollViewport: DEFAULT_PIANO_ROLL_VIEWPORT,

  isPlaying: false,

  // === Viewport ===
  setTimelineViewport: (viewport) => {
    set((state) => ({
      timelineViewport: { ...state.timelineViewport, ...viewport },
    }));
  },

  zoomTimeline: (factor, centerTick) => {
    const { timelineViewport } = get();
    const currentCenter = centerTick ?? (timelineViewport.startTick + timelineViewport.endTick) / 2;
    const currentRange = timelineViewport.endTick - timelineViewport.startTick;
    const newRange = currentRange / factor;

    const minRange = TICKS_PER_BEAT * 4; // Minimum 1 bar visible
    const maxRange = TICKS_PER_BEAT * 256; // Maximum 64 bars visible

    const clampedRange = Math.max(minRange, Math.min(maxRange, newRange));
    const halfRange = clampedRange / 2;

    set({
      timelineViewport: {
        ...timelineViewport,
        startTick: Math.max(0, currentCenter - halfRange),
        endTick: currentCenter + halfRange,
        zoom: timelineViewport.zoom * factor,
      },
    });
  },

  scrollTimeline: (deltaTick) => {
    const { timelineViewport } = get();
    const range = timelineViewport.endTick - timelineViewport.startTick;
    const newStart = Math.max(0, timelineViewport.startTick + deltaTick);

    set({
      timelineViewport: {
        ...timelineViewport,
        startTick: newStart,
        endTick: newStart + range,
      },
    });
  },

  scrollTimelineVertical: (deltaPixels) => {
    set((state) => ({
      timelineViewport: {
        ...state.timelineViewport,
        verticalScroll: Math.max(0, state.timelineViewport.verticalScroll + deltaPixels),
      },
    }));
  },

  // === Snap ===
  setSnapEnabled: (enabled) => {
    set((state) => ({
      snapSettings: { ...state.snapSettings, enabled },
    }));
  },

  setSnapValue: (value) => {
    set((state) => ({
      snapSettings: { ...state.snapSettings, value },
    }));
  },

  // === Tools ===
  setTool: (tool) => {
    set({ currentTool: tool });
  },

  // === Panels ===
  setActivePanel: (panel) => {
    set({ activePanel: panel });
  },

  togglePianoRoll: () => {
    set((state) => ({ isPianoRollOpen: !state.isPianoRollOpen }));
  },

  toggleMixer: () => {
    set((state) => ({ isMixerOpen: !state.isMixerOpen }));
  },

  toggleBeatboxPanel: () => {
    set((state) => ({ isBeatboxPanelOpen: !state.isBeatboxPanelOpen }));
  },

  openPianoRoll: (trackId, clipId) => {
    set({
      isPianoRollOpen: true,
      pianoRollTrackId: trackId,
      pianoRollClipId: clipId,
    });
  },

  closePianoRoll: () => {
    set({
      isPianoRollOpen: false,
      pianoRollTrackId: null,
      pianoRollClipId: null,
    });
  },

  // === Selection ===
  selectNotes: (noteIds) => {
    set({ selectedNoteIds: noteIds });
  },

  addToNoteSelection: (noteIds) => {
    set((state) => ({
      selectedNoteIds: [...new Set([...state.selectedNoteIds, ...noteIds])],
    }));
  },

  clearNoteSelection: () => {
    set({ selectedNoteIds: [] });
  },

  // === Piano Roll Viewport ===
  setPianoRollViewport: (viewport) => {
    set((state) => ({
      pianoRollViewport: { ...state.pianoRollViewport, ...viewport },
    }));
  },

  zoomPianoRoll: (factor) => {
    const { pianoRollViewport } = get();
    const currentCenter = (pianoRollViewport.startTick + pianoRollViewport.endTick) / 2;
    const currentRange = pianoRollViewport.endTick - pianoRollViewport.startTick;
    const newRange = currentRange / factor;

    const minRange = TICKS_PER_BEAT; // Minimum 1 beat visible
    const maxRange = TICKS_PER_BEAT * 64; // Maximum 16 bars visible

    const clampedRange = Math.max(minRange, Math.min(maxRange, newRange));
    const halfRange = clampedRange / 2;

    set({
      pianoRollViewport: {
        ...pianoRollViewport,
        startTick: Math.max(0, currentCenter - halfRange),
        endTick: currentCenter + halfRange,
        zoom: pianoRollViewport.zoom * factor,
      },
    });
  },

  scrollPianoRoll: (deltaTick) => {
    const { pianoRollViewport } = get();
    const range = pianoRollViewport.endTick - pianoRollViewport.startTick;
    const newStart = Math.max(0, pianoRollViewport.startTick + deltaTick);

    set({
      pianoRollViewport: {
        ...pianoRollViewport,
        startTick: newStart,
        endTick: newStart + range,
      },
    });
  },

  scrollPianoRollVertical: (deltaPitch) => {
    const { pianoRollViewport } = get();
    const range = pianoRollViewport.topPitch - pianoRollViewport.bottomPitch;

    let newTop = pianoRollViewport.topPitch + deltaPitch;
    let newBottom = pianoRollViewport.bottomPitch + deltaPitch;

    // Clamp to valid MIDI range
    if (newTop > 127) {
      newTop = 127;
      newBottom = 127 - range;
    }
    if (newBottom < 0) {
      newBottom = 0;
      newTop = range;
    }

    set({
      pianoRollViewport: {
        ...pianoRollViewport,
        topPitch: newTop,
        bottomPitch: newBottom,
      },
    });
  },

  // === Misc ===
  setIsPlaying: (isPlaying) => {
    set({ isPlaying });
  },
}));
