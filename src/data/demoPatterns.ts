/**
 * Demo Patterns
 * Pre-recorded drum patterns for demonstration when backend is unavailable
 */

import { DrumEvent } from '../hooks/useWebSocket';

export interface DemoPattern {
  name: string;
  bpm: number;
  events: DrumEvent[];
  duration: number; // in milliseconds
}

// Basic four-on-the-floor pattern
export const basicBeat: DemoPattern = {
  name: 'Basic Beat',
  bpm: 120,
  duration: 2000,
  events: [
    { drum_class: 'kick', confidence: 0.95, midi_note: 36, velocity: 100, timestamp: 0 },
    { drum_class: 'hihat', confidence: 0.88, midi_note: 42, velocity: 80, timestamp: 0 },
    { drum_class: 'hihat', confidence: 0.85, midi_note: 42, velocity: 70, timestamp: 250 },
    { drum_class: 'snare', confidence: 0.92, midi_note: 38, velocity: 95, timestamp: 500 },
    { drum_class: 'hihat', confidence: 0.87, midi_note: 42, velocity: 75, timestamp: 500 },
    { drum_class: 'hihat', confidence: 0.84, midi_note: 42, velocity: 70, timestamp: 750 },
    { drum_class: 'kick', confidence: 0.93, midi_note: 36, velocity: 100, timestamp: 1000 },
    { drum_class: 'hihat', confidence: 0.86, midi_note: 42, velocity: 80, timestamp: 1000 },
    { drum_class: 'hihat', confidence: 0.82, midi_note: 42, velocity: 65, timestamp: 1250 },
    { drum_class: 'snare', confidence: 0.91, midi_note: 38, velocity: 95, timestamp: 1500 },
    { drum_class: 'hihat', confidence: 0.88, midi_note: 42, velocity: 75, timestamp: 1500 },
    { drum_class: 'hihat', confidence: 0.85, midi_note: 42, velocity: 70, timestamp: 1750 },
  ],
};

// Hip-hop style pattern
export const hipHopBeat: DemoPattern = {
  name: 'Hip-Hop',
  bpm: 90,
  duration: 2667,
  events: [
    { drum_class: 'kick', confidence: 0.96, midi_note: 36, velocity: 110, timestamp: 0 },
    { drum_class: 'hihat', confidence: 0.85, midi_note: 42, velocity: 70, timestamp: 167 },
    { drum_class: 'hihat', confidence: 0.82, midi_note: 42, velocity: 65, timestamp: 333 },
    { drum_class: 'snare', confidence: 0.94, midi_note: 38, velocity: 100, timestamp: 667 },
    { drum_class: 'hihat', confidence: 0.83, midi_note: 42, velocity: 68, timestamp: 833 },
    { drum_class: 'kick', confidence: 0.88, midi_note: 36, velocity: 90, timestamp: 1000 },
    { drum_class: 'hihat', confidence: 0.81, midi_note: 42, velocity: 65, timestamp: 1167 },
    { drum_class: 'kick', confidence: 0.91, midi_note: 36, velocity: 95, timestamp: 1333 },
    { drum_class: 'hihat', confidence: 0.84, midi_note: 42, velocity: 70, timestamp: 1500 },
    { drum_class: 'snare', confidence: 0.95, midi_note: 38, velocity: 105, timestamp: 2000 },
    { drum_class: 'hihat', confidence: 0.86, midi_note: 42, velocity: 72, timestamp: 2167 },
    { drum_class: 'hihat', confidence: 0.80, midi_note: 42, velocity: 60, timestamp: 2333 },
  ],
};

// Drum and bass pattern
export const dnbBeat: DemoPattern = {
  name: 'Drum & Bass',
  bpm: 174,
  duration: 1379,
  events: [
    { drum_class: 'kick', confidence: 0.97, midi_note: 36, velocity: 115, timestamp: 0 },
    { drum_class: 'hihat', confidence: 0.88, midi_note: 42, velocity: 75, timestamp: 86 },
    { drum_class: 'hihat', confidence: 0.85, midi_note: 42, velocity: 70, timestamp: 172 },
    { drum_class: 'snare', confidence: 0.93, midi_note: 38, velocity: 100, timestamp: 345 },
    { drum_class: 'hihat', confidence: 0.82, midi_note: 42, velocity: 68, timestamp: 431 },
    { drum_class: 'hihat', confidence: 0.84, midi_note: 42, velocity: 72, timestamp: 517 },
    { drum_class: 'kick', confidence: 0.89, midi_note: 36, velocity: 95, timestamp: 603 },
    { drum_class: 'hihat', confidence: 0.81, midi_note: 42, velocity: 65, timestamp: 690 },
    { drum_class: 'kick', confidence: 0.92, midi_note: 36, velocity: 100, timestamp: 776 },
    { drum_class: 'hihat', confidence: 0.86, midi_note: 42, velocity: 74, timestamp: 862 },
    { drum_class: 'snare', confidence: 0.94, midi_note: 38, velocity: 105, timestamp: 1034 },
    { drum_class: 'hihat', confidence: 0.83, midi_note: 42, velocity: 70, timestamp: 1121 },
    { drum_class: 'hihat', confidence: 0.80, midi_note: 42, velocity: 62, timestamp: 1207 },
  ],
};

// Breakbeat pattern
export const breakbeat: DemoPattern = {
  name: 'Breakbeat',
  bpm: 130,
  duration: 1846,
  events: [
    { drum_class: 'kick', confidence: 0.95, midi_note: 36, velocity: 105, timestamp: 0 },
    { drum_class: 'hihat', confidence: 0.87, midi_note: 42, velocity: 78, timestamp: 115 },
    { drum_class: 'snare', confidence: 0.90, midi_note: 38, velocity: 92, timestamp: 231 },
    { drum_class: 'hihat', confidence: 0.84, midi_note: 42, velocity: 72, timestamp: 346 },
    { drum_class: 'kick', confidence: 0.91, midi_note: 36, velocity: 98, timestamp: 462 },
    { drum_class: 'hihat', confidence: 0.82, midi_note: 42, velocity: 68, timestamp: 577 },
    { drum_class: 'snare', confidence: 0.93, midi_note: 38, velocity: 100, timestamp: 692 },
    { drum_class: 'kick', confidence: 0.88, midi_note: 36, velocity: 90, timestamp: 808 },
    { drum_class: 'hihat', confidence: 0.85, midi_note: 42, velocity: 75, timestamp: 923 },
    { drum_class: 'kick', confidence: 0.94, midi_note: 36, velocity: 102, timestamp: 1038 },
    { drum_class: 'hihat', confidence: 0.81, midi_note: 42, velocity: 65, timestamp: 1154 },
    { drum_class: 'snare', confidence: 0.92, midi_note: 38, velocity: 97, timestamp: 1269 },
    { drum_class: 'hihat', confidence: 0.83, midi_note: 42, velocity: 70, timestamp: 1385 },
    { drum_class: 'kick', confidence: 0.89, midi_note: 36, velocity: 94, timestamp: 1500 },
    { drum_class: 'snare', confidence: 0.86, midi_note: 38, velocity: 88, timestamp: 1615 },
    { drum_class: 'hihat', confidence: 0.80, midi_note: 42, velocity: 62, timestamp: 1731 },
  ],
};

// All demo patterns
export const demoPatterns: DemoPattern[] = [
  basicBeat,
  hipHopBeat,
  dnbBeat,
  breakbeat,
];

// Get a random pattern
export function getRandomPattern(): DemoPattern {
  return demoPatterns[Math.floor(Math.random() * demoPatterns.length)];
}

// Generate pattern events with current timestamps
export function generatePatternEvents(pattern: DemoPattern): DrumEvent[] {
  const now = Date.now();
  return pattern.events.map((event) => ({
    ...event,
    timestamp: now + event.timestamp,
  }));
}
