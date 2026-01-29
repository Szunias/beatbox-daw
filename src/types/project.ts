/**
 * DAW Project Types
 * Core type definitions for the BeatBox DAW
 */

// === MIDI Types ===
export interface MidiNote {
  id: string;
  pitch: number;      // MIDI note number (0-127)
  velocity: number;   // Note velocity (0-127)
  startTick: number;  // Start position in ticks
  duration: number;   // Duration in ticks
}

export interface MidiClipData {
  notes: MidiNote[];
  duration: number;   // Total clip duration in ticks
}

// === Clip Types ===
export type ClipType = 'midi' | 'audio' | 'beatbox';

export interface BaseClip {
  id: string;
  name: string;
  type: ClipType;
  startTick: number;      // Position on timeline in ticks
  duration: number;       // Duration in ticks
  color: string;          // Display color
  muted: boolean;
}

export interface MidiClip extends BaseClip {
  type: 'midi';
  data: MidiClipData;
}

export interface AudioClip extends BaseClip {
  type: 'audio';
  audioFilePath: string;
  waveformData?: number[];  // Cached waveform for display
}

export interface BeatboxClip extends BaseClip {
  type: 'beatbox';
  data: MidiClipData;       // Converted to MIDI internally
  originalEvents: DrumEvent[];
}

export type Clip = MidiClip | AudioClip | BeatboxClip;

// === Track Types ===
export type TrackType = 'midi' | 'audio' | 'drum' | 'master';

export interface TrackEffectSlot {
  id: string;
  pluginId: string | null;  // VST plugin ID or null if empty
  parameters: Record<string, number>;
  bypassed: boolean;
}

export interface BaseTrack {
  id: string;
  name: string;
  type: TrackType;
  color: string;
  volume: number;       // 0-1
  pan: number;          // -1 to 1 (left to right)
  muted: boolean;
  solo: boolean;
  armed: boolean;       // Record armed
  clips: Clip[];
  effectSlots: TrackEffectSlot[];
  instrumentId: string | null;  // VST instrument ID
}

export interface MidiTrack extends BaseTrack {
  type: 'midi';
  midiChannel: number;  // 1-16
}

export interface AudioTrack extends BaseTrack {
  type: 'audio';
  inputSource: string;  // Audio input device/channel
}

export interface DrumTrack extends BaseTrack {
  type: 'drum';
  drumMap: DrumMapping;
}

export interface MasterTrack extends BaseTrack {
  type: 'master';
}

export type Track = MidiTrack | AudioTrack | DrumTrack | MasterTrack;

// === Drum Mapping ===
export interface DrumMapping {
  kick: number;
  snare: number;
  hihat: number;
  clap: number;
  tom: number;
  [key: string]: number;
}

export const DEFAULT_DRUM_MAPPING: DrumMapping = {
  kick: 36,
  snare: 38,
  hihat: 42,
  clap: 39,
  tom: 45,
};

// === Transport Types ===
export interface LoopRegion {
  enabled: boolean;
  startTick: number;
  endTick: number;
}

export type TransportState = 'stopped' | 'playing' | 'paused' | 'recording';

export interface TransportSettings {
  bpm: number;
  timeSignatureNumerator: number;
  timeSignatureDenominator: number;
  clickEnabled: boolean;
  clickVolume: number;
}

// === Project Types ===
export interface Project {
  id: string;
  name: string;
  createdAt: number;
  modifiedAt: number;
  bpm: number;
  timeSignatureNumerator: number;
  timeSignatureDenominator: number;
  tracks: Track[];
  masterTrack: MasterTrack;
}

// === Timeline Types ===
export interface TimelineViewport {
  startTick: number;
  endTick: number;
  zoom: number;           // Pixels per tick
  verticalScroll: number; // Vertical scroll position in pixels
}

export interface SnapSettings {
  enabled: boolean;
  value: SnapValue;
}

export type SnapValue = 'none' | '1/1' | '1/2' | '1/4' | '1/8' | '1/16' | '1/32';

// === Selection Types ===
export interface Selection {
  trackIds: string[];
  clipIds: string[];
  noteIds: string[];
}

// === Event Types (from BeatBox detector) ===
export interface DrumEvent {
  drum_class: string;
  confidence: number;
  midi_note: number;
  velocity: number;
  timestamp: number;
}

// === Utility Functions ===

export const TICKS_PER_BEAT = 480;  // Standard MIDI resolution

export function ticksToBeats(ticks: number): number {
  return ticks / TICKS_PER_BEAT;
}

export function beatsToTicks(beats: number): number {
  return beats * TICKS_PER_BEAT;
}

export function ticksToSeconds(ticks: number, bpm: number): number {
  return (ticks / TICKS_PER_BEAT) * (60 / bpm);
}

export function secondsToTicks(seconds: number, bpm: number): number {
  return (seconds / 60) * bpm * TICKS_PER_BEAT;
}

export function ticksToMeasures(ticks: number, numerator: number): number {
  return ticks / (TICKS_PER_BEAT * numerator);
}

export function measuresToTicks(measures: number, numerator: number): number {
  return measures * TICKS_PER_BEAT * numerator;
}

export function snapTickToGrid(tick: number, snapValue: SnapValue, numerator: number): number {
  if (snapValue === 'none') return tick;

  const snapTicks = getSnapTicks(snapValue, numerator);
  return Math.round(tick / snapTicks) * snapTicks;
}

export function getSnapTicks(snapValue: SnapValue, numerator: number = 4): number {
  switch (snapValue) {
    case 'none': return 1;
    case '1/1': return TICKS_PER_BEAT * numerator;
    case '1/2': return TICKS_PER_BEAT * (numerator / 2);
    case '1/4': return TICKS_PER_BEAT;
    case '1/8': return TICKS_PER_BEAT / 2;
    case '1/16': return TICKS_PER_BEAT / 4;
    case '1/32': return TICKS_PER_BEAT / 8;
    default: return TICKS_PER_BEAT;
  }
}

// === Factory Functions ===

let idCounter = 0;

export function generateId(prefix: string = 'id'): string {
  return `${prefix}_${Date.now()}_${++idCounter}`;
}

export function createMidiNote(
  pitch: number,
  velocity: number,
  startTick: number,
  duration: number
): MidiNote {
  return {
    id: generateId('note'),
    pitch,
    velocity,
    startTick,
    duration,
  };
}

export function createMidiClip(
  name: string,
  startTick: number,
  duration: number,
  notes: MidiNote[] = [],
  color: string = '#4ade80'
): MidiClip {
  return {
    id: generateId('clip'),
    name,
    type: 'midi',
    startTick,
    duration,
    color,
    muted: false,
    data: {
      notes,
      duration,
    },
  };
}

export function createMidiTrack(
  name: string,
  color: string = '#4ade80',
  midiChannel: number = 1
): MidiTrack {
  return {
    id: generateId('track'),
    name,
    type: 'midi',
    color,
    volume: 0.8,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [],
    effectSlots: [],
    instrumentId: null,
    midiChannel,
  };
}

export function createDrumTrack(
  name: string = 'Drums',
  color: string = '#e94560'
): DrumTrack {
  return {
    id: generateId('track'),
    name,
    type: 'drum',
    color,
    volume: 0.8,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [],
    effectSlots: [],
    instrumentId: null,
    drumMap: { ...DEFAULT_DRUM_MAPPING },
  };
}

export function createAudioTrack(
  name: string,
  color: string = '#60a5fa'
): AudioTrack {
  return {
    id: generateId('track'),
    name,
    type: 'audio',
    color,
    volume: 0.8,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [],
    effectSlots: [],
    instrumentId: null,
    inputSource: 'default',
  };
}

export function createMasterTrack(): MasterTrack {
  return {
    id: generateId('master'),
    name: 'Master',
    type: 'master',
    color: '#fbbf24',
    volume: 1.0,
    pan: 0,
    muted: false,
    solo: false,
    armed: false,
    clips: [],
    effectSlots: [],
    instrumentId: null,
  };
}

export function createProject(name: string = 'Untitled Project'): Project {
  return {
    id: generateId('project'),
    name,
    createdAt: Date.now(),
    modifiedAt: Date.now(),
    bpm: 120,
    timeSignatureNumerator: 4,
    timeSignatureDenominator: 4,
    tracks: [
      createDrumTrack('BeatBox Drums'),
    ],
    masterTrack: createMasterTrack(),
  };
}
