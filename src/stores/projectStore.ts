/**
 * Project Store
 * Zustand store for managing the DAW project state
 * Syncs with backend via WebSocket
 */

import { create } from 'zustand';
import {
  Project,
  Track,
  Clip,
  MidiNote,
  MidiClip,
  AudioClip,
  createProject,
  createMidiTrack,
  createDrumTrack,
  createAudioTrack,
  createAudioClip,
  createMidiClip,
  createMidiNote,
  generateId,
  DrumEvent,
  TICKS_PER_BEAT,
  secondsToTicks,
} from '../types/project';

// WebSocket send function type
type WebSocketSendFn = (type: string, data?: Record<string, unknown>) => void;

interface ProjectState {
  project: Project;
  selectedTrackId: string | null;
  selectedClipId: string | null;
  clipboardClips: Clip[];
  clipboardNotes: MidiNote[];
  undoStack: Project[];
  redoStack: Project[];

  // WebSocket connection
  _wsSend: WebSocketSendFn | null;
  _isConnected: boolean;
}

interface ProjectActions {
  // Project
  newProject: (name?: string) => void;
  setProjectName: (name: string) => void;
  setBpm: (bpm: number) => void;
  setTimeSignature: (numerator: number, denominator: number) => void;

  // Tracks
  addTrack: (type: 'midi' | 'audio' | 'drum', name?: string) => string;
  removeTrack: (trackId: string) => void;
  updateTrack: (trackId: string, updates: Partial<Track>) => void;
  reorderTrack: (trackId: string, newIndex: number) => void;
  setTrackVolume: (trackId: string, volume: number) => void;
  setTrackPan: (trackId: string, pan: number) => void;
  toggleTrackMute: (trackId: string) => void;
  toggleTrackSolo: (trackId: string) => void;
  toggleTrackArmed: (trackId: string) => void;

  // Clips
  addClip: (trackId: string, clip: Clip) => void;
  removeClip: (trackId: string, clipId: string) => void;
  updateClip: (trackId: string, clipId: string, updates: Partial<Clip>) => void;
  moveClip: (trackId: string, clipId: string, newStartTick: number) => void;
  resizeClip: (trackId: string, clipId: string, newDuration: number) => void;
  duplicateClip: (trackId: string, clipId: string) => void;

  // MIDI Notes
  addNote: (trackId: string, clipId: string, note: MidiNote) => void;
  removeNote: (trackId: string, clipId: string, noteId: string) => void;
  updateNote: (trackId: string, clipId: string, noteId: string, updates: Partial<MidiNote>) => void;
  moveNote: (trackId: string, clipId: string, noteId: string, newPitch: number, newStartTick: number) => void;

  // BeatBox Integration
  addBeatboxEvents: (trackId: string, events: DrumEvent[], startTick: number) => void;

  // Audio Recording Integration
  addAudioClipFromRecording: (
    trackId: string,
    audioFilePath: string,
    startTick: number,
    durationSeconds: number
  ) => void;

  // Selection
  selectTrack: (trackId: string | null) => void;
  selectClip: (clipId: string | null) => void;

  // Clipboard
  copyClips: (clipIds: string[]) => void;
  cutClips: (trackId: string, clipIds: string[]) => void;
  pasteClips: (trackId: string, startTick: number) => void;
  copyNotes: (noteIds: string[], trackId: string, clipId: string) => void;
  pasteNotes: (trackId: string, clipId: string, startTick: number) => void;

  // Undo/Redo
  undo: () => void;
  redo: () => void;
  saveState: () => void;

  // Getters
  getTrack: (trackId: string) => Track | undefined;
  getClip: (trackId: string, clipId: string) => Clip | undefined;

  // WebSocket sync
  setWebSocket: (sendFn: WebSocketSendFn | null, isConnected: boolean) => void;
}

const MAX_UNDO_STACK = 50;

// We need to install immer middleware
// For now, let's create the store without immer
export const useProjectStore = create<ProjectState & ProjectActions>()((set, get) => ({
  // Initial state
  project: createProject('New Project'),
  selectedTrackId: null,
  selectedClipId: null,
  clipboardClips: [],
  clipboardNotes: [],
  undoStack: [],
  redoStack: [],

  _wsSend: null,
  _isConnected: false,

  // === Project Actions ===
  newProject: (name = 'New Project') => {
    set({
      project: createProject(name),
      selectedTrackId: null,
      selectedClipId: null,
      undoStack: [],
      redoStack: [],
    });
  },

  setProjectName: (name) => {
    set((state) => ({
      project: {
        ...state.project,
        name,
        modifiedAt: Date.now(),
      },
    }));
  },

  setBpm: (bpm) => {
    const { _wsSend, _isConnected } = get();
    const newBpm = Math.max(20, Math.min(300, bpm));

    // Send to backend if connected
    if (_isConnected && _wsSend) {
      _wsSend('set_bpm', { bpm: newBpm });
    }

    set((state) => ({
      project: {
        ...state.project,
        bpm: newBpm,
        modifiedAt: Date.now(),
      },
    }));
  },

  setTimeSignature: (numerator, denominator) => {
    set((state) => ({
      project: {
        ...state.project,
        timeSignatureNumerator: numerator,
        timeSignatureDenominator: denominator,
        modifiedAt: Date.now(),
      },
    }));
  },

  // === Track Actions ===
  addTrack: (type, name) => {
    let newTrack: Track;
    const trackCount = get().project.tracks.length + 1;
    const defaultName = name || `Track ${trackCount}`;

    switch (type) {
      case 'midi':
        newTrack = createMidiTrack(defaultName);
        break;
      case 'audio':
        newTrack = createAudioTrack(defaultName);
        break;
      case 'drum':
        newTrack = createDrumTrack(defaultName);
        break;
      default:
        newTrack = createMidiTrack(defaultName);
    }

    get().saveState();
    set((state) => ({
      project: {
        ...state.project,
        tracks: [...state.project.tracks, newTrack],
        modifiedAt: Date.now(),
      },
    }));

    return newTrack.id;
  },

  removeTrack: (trackId) => {
    get().saveState();
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.filter((t) => t.id !== trackId),
        modifiedAt: Date.now(),
      },
      selectedTrackId: state.selectedTrackId === trackId ? null : state.selectedTrackId,
    }));
  },

  updateTrack: (trackId, updates) => {
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId ? { ...t, ...updates } as Track : t
        ),
        modifiedAt: Date.now(),
      },
    }));
  },

  reorderTrack: (trackId, newIndex) => {
    const tracks = [...get().project.tracks];
    const currentIndex = tracks.findIndex((t) => t.id === trackId);
    if (currentIndex === -1 || newIndex < 0 || newIndex >= tracks.length) return;

    const [track] = tracks.splice(currentIndex, 1);
    tracks.splice(newIndex, 0, track);

    set((state) => ({
      project: {
        ...state.project,
        tracks,
        modifiedAt: Date.now(),
      },
    }));
  },

  setTrackVolume: (trackId, volume) => {
    get().updateTrack(trackId, { volume: Math.max(0, Math.min(1, volume)) });
  },

  setTrackPan: (trackId, pan) => {
    get().updateTrack(trackId, { pan: Math.max(-1, Math.min(1, pan)) });
  },

  toggleTrackMute: (trackId) => {
    const track = get().getTrack(trackId);
    if (track) {
      get().updateTrack(trackId, { muted: !track.muted });
    }
  },

  toggleTrackSolo: (trackId) => {
    const track = get().getTrack(trackId);
    if (track) {
      get().updateTrack(trackId, { solo: !track.solo });
    }
  },

  toggleTrackArmed: (trackId) => {
    const track = get().getTrack(trackId);
    if (track) {
      const newArmedState = !track.armed;

      // If arming this track, disarm all others (only one can be armed at a time)
      if (newArmedState) {
        const state = get();
        state.project.tracks.forEach((t) => {
          if (t.id !== trackId && t.armed) {
            get().updateTrack(t.id, { armed: false });
          }
        });
      }

      // Sync with backend
      const { _wsSend, _isConnected } = get();
      if (_isConnected && _wsSend) {
        _wsSend('set_track_armed', { track_id: trackId, armed: newArmedState });
      }

      get().updateTrack(trackId, { armed: newArmedState });
    }
  },

  // === Clip Actions ===
  addClip: (trackId, clip) => {
    get().saveState();
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId
            ? { ...t, clips: [...t.clips, clip] } as Track
            : t
        ),
        modifiedAt: Date.now(),
      },
    }));
  },

  removeClip: (trackId, clipId) => {
    get().saveState();
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId
            ? { ...t, clips: t.clips.filter((c) => c.id !== clipId) } as Track
            : t
        ),
        modifiedAt: Date.now(),
      },
      selectedClipId: state.selectedClipId === clipId ? null : state.selectedClipId,
    }));
  },

  updateClip: (trackId, clipId, updates) => {
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId
            ? {
                ...t,
                clips: t.clips.map((c) =>
                  c.id === clipId ? { ...c, ...updates } as Clip : c
                ),
              } as Track
            : t
        ),
        modifiedAt: Date.now(),
      },
    }));
  },

  moveClip: (trackId, clipId, newStartTick) => {
    get().updateClip(trackId, clipId, { startTick: Math.max(0, newStartTick) });
  },

  resizeClip: (trackId, clipId, newDuration) => {
    get().updateClip(trackId, clipId, { duration: Math.max(TICKS_PER_BEAT / 4, newDuration) });
  },

  duplicateClip: (trackId, clipId) => {
    const clip = get().getClip(trackId, clipId);
    if (!clip) return;

    const newClip = {
      ...clip,
      id: generateId('clip'),
      startTick: clip.startTick + clip.duration,
    };

    get().addClip(trackId, newClip);
  },

  // === MIDI Note Actions ===
  addNote: (trackId, clipId, note) => {
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId
            ? {
                ...t,
                clips: t.clips.map((c) => {
                  if (c.id !== clipId || c.type !== 'midi') return c;
                  const midiClip = c as MidiClip;
                  return {
                    ...midiClip,
                    data: {
                      ...midiClip.data,
                      notes: [...midiClip.data.notes, note],
                    },
                  } as Clip;
                }),
              } as Track
            : t
        ),
        modifiedAt: Date.now(),
      },
    }));
  },

  removeNote: (trackId, clipId, noteId) => {
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId
            ? {
                ...t,
                clips: t.clips.map((c) => {
                  if (c.id !== clipId || c.type !== 'midi') return c;
                  const midiClip = c as MidiClip;
                  return {
                    ...midiClip,
                    data: {
                      ...midiClip.data,
                      notes: midiClip.data.notes.filter((n) => n.id !== noteId),
                    },
                  } as Clip;
                }),
              } as Track
            : t
        ),
        modifiedAt: Date.now(),
      },
    }));
  },

  updateNote: (trackId, clipId, noteId, updates) => {
    set((state) => ({
      project: {
        ...state.project,
        tracks: state.project.tracks.map((t) =>
          t.id === trackId
            ? {
                ...t,
                clips: t.clips.map((c) => {
                  if (c.id !== clipId || c.type !== 'midi') return c;
                  const midiClip = c as MidiClip;
                  return {
                    ...midiClip,
                    data: {
                      ...midiClip.data,
                      notes: midiClip.data.notes.map((n) =>
                        n.id === noteId ? { ...n, ...updates } : n
                      ),
                    },
                  } as Clip;
                }),
              } as Track
            : t
        ),
        modifiedAt: Date.now(),
      },
    }));
  },

  moveNote: (trackId, clipId, noteId, newPitch, newStartTick) => {
    get().updateNote(trackId, clipId, noteId, {
      pitch: Math.max(0, Math.min(127, newPitch)),
      startTick: Math.max(0, newStartTick),
    });
  },

  // === BeatBox Integration ===
  addBeatboxEvents: (trackId, events, startTick) => {
    if (events.length === 0) return;

    const bpm = get().project.bpm;
    const notes: MidiNote[] = events.map((event) => ({
      id: generateId('note'),
      pitch: event.midi_note,
      velocity: event.velocity,
      startTick: startTick + secondsToTicks(event.timestamp, bpm),
      duration: TICKS_PER_BEAT / 4, // Default to 16th note
    }));

    const duration = Math.max(
      TICKS_PER_BEAT * 4, // Minimum 1 bar
      ...notes.map((n) => n.startTick + n.duration)
    );

    const clip = createMidiClip(
      `BeatBox ${new Date().toLocaleTimeString()}`,
      startTick,
      duration,
      notes,
      '#e94560'
    );

    get().addClip(trackId, clip);
  },

  // === Audio Recording Integration ===
  addAudioClipFromRecording: (trackId, audioFilePath, startTick, durationSeconds) => {
    const bpm = get().project.bpm;

    // Convert duration from seconds to ticks
    const durationTicks = secondsToTicks(durationSeconds, bpm);

    // Create the audio clip
    const clip = createAudioClip(
      `Recording ${new Date().toLocaleTimeString()}`,
      startTick,
      Math.max(TICKS_PER_BEAT, durationTicks), // Minimum 1 beat
      audioFilePath,
      '#60a5fa' // Blue color for audio clips
    );

    get().addClip(trackId, clip);
  },

  // === Selection ===
  selectTrack: (trackId) => {
    set({ selectedTrackId: trackId });
  },

  selectClip: (clipId) => {
    set({ selectedClipId: clipId });
  },

  // === Clipboard ===
  copyClips: (clipIds) => {
    const clips: Clip[] = [];
    get().project.tracks.forEach((track) => {
      track.clips.forEach((clip) => {
        if (clipIds.includes(clip.id)) {
          clips.push({ ...clip, id: generateId('clip') });
        }
      });
    });
    set({ clipboardClips: clips });
  },

  cutClips: (trackId, clipIds) => {
    get().copyClips(clipIds);
    clipIds.forEach((clipId) => get().removeClip(trackId, clipId));
  },

  pasteClips: (trackId, startTick) => {
    const clips = get().clipboardClips;
    if (clips.length === 0) return;

    const minStartTick = Math.min(...clips.map((c) => c.startTick));
    clips.forEach((clip) => {
      const newClip = {
        ...clip,
        id: generateId('clip'),
        startTick: startTick + (clip.startTick - minStartTick),
      };
      get().addClip(trackId, newClip);
    });
  },

  copyNotes: (noteIds, trackId, clipId) => {
    const clip = get().getClip(trackId, clipId);
    if (!clip || clip.type !== 'midi') return;

    const midiClip = clip as MidiClip;
    const notes = midiClip.data.notes.filter((n) => noteIds.includes(n.id));
    set({ clipboardNotes: notes.map((n) => ({ ...n, id: generateId('note') })) });
  },

  pasteNotes: (trackId, clipId, startTick) => {
    const notes = get().clipboardNotes;
    if (notes.length === 0) return;

    const minStartTick = Math.min(...notes.map((n) => n.startTick));
    notes.forEach((note) => {
      const newNote = createMidiNote(
        note.pitch,
        note.velocity,
        startTick + (note.startTick - minStartTick),
        note.duration
      );
      get().addNote(trackId, clipId, newNote);
    });
  },

  // === Undo/Redo ===
  saveState: () => {
    const currentProject = get().project;
    set((state) => ({
      undoStack: [...state.undoStack.slice(-MAX_UNDO_STACK + 1), currentProject],
      redoStack: [],
    }));
  },

  undo: () => {
    const undoStack = get().undoStack;
    if (undoStack.length === 0) return;

    const previousProject = undoStack[undoStack.length - 1];
    set((state) => ({
      project: previousProject,
      undoStack: state.undoStack.slice(0, -1),
      redoStack: [...state.redoStack, state.project],
    }));
  },

  redo: () => {
    const redoStack = get().redoStack;
    if (redoStack.length === 0) return;

    const nextProject = redoStack[redoStack.length - 1];
    set((state) => ({
      project: nextProject,
      undoStack: [...state.undoStack, state.project],
      redoStack: state.redoStack.slice(0, -1),
    }));
  },

  // === Getters ===
  getTrack: (trackId) => {
    return get().project.tracks.find((t) => t.id === trackId);
  },

  getClip: (trackId, clipId) => {
    const track = get().getTrack(trackId);
    return track?.clips.find((c) => c.id === clipId);
  },

  // === WebSocket Sync ===
  setWebSocket: (sendFn, isConnected) => {
    set({
      _wsSend: sendFn,
      _isConnected: isConnected,
    });
  },
}));
