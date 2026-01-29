/**
 * useAudioPreview Hook
 * Simple Web Audio synthesizer for previewing notes in piano roll
 */

import { useCallback, useRef, useEffect } from 'react';

// MIDI note to frequency conversion
const midiToFrequency = (midiNote: number): number => {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
};

interface UseAudioPreviewReturn {
  playNote: (pitch: number, duration?: number, velocity?: number) => void;
  stopNote: (pitch: number) => void;
  stopAll: () => void;
}

export const useAudioPreview = (): UseAudioPreviewReturn => {
  const audioContextRef = useRef<AudioContext | null>(null);
  const activeNotesRef = useRef<Map<number, { oscillator: OscillatorNode; gain: GainNode }>>(new Map());

  // Initialize audio context on first interaction
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    // Resume if suspended (browsers require user interaction)
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  const playNote = useCallback((pitch: number, duration: number = 0.3, velocity: number = 100) => {
    const ctx = getAudioContext();

    // Stop any existing note at this pitch
    if (activeNotesRef.current.has(pitch)) {
      stopNote(pitch);
    }

    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    // Set frequency from MIDI note
    oscillator.frequency.value = midiToFrequency(pitch);
    oscillator.type = 'triangle'; // Softer sound than sine

    // Set volume based on velocity (0-127)
    const volume = (velocity / 127) * 0.3; // Max 30% volume
    gainNode.gain.setValueAtTime(volume, ctx.currentTime);

    // Connect nodes
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    // Start playing
    oscillator.start(ctx.currentTime);

    // Store reference
    activeNotesRef.current.set(pitch, { oscillator, gain: gainNode });

    // Auto-stop after duration (with fade out)
    if (duration > 0) {
      const fadeTime = Math.min(0.1, duration * 0.3);
      gainNode.gain.setValueAtTime(volume, ctx.currentTime + duration - fadeTime);
      gainNode.gain.linearRampToValueAtTime(0, ctx.currentTime + duration);

      setTimeout(() => {
        stopNote(pitch);
      }, duration * 1000);
    }
  }, [getAudioContext]);

  const stopNote = useCallback((pitch: number) => {
    const note = activeNotesRef.current.get(pitch);
    if (note) {
      const ctx = audioContextRef.current;
      if (ctx) {
        // Quick fade out to avoid clicks
        note.gain.gain.setValueAtTime(note.gain.gain.value, ctx.currentTime);
        note.gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.05);

        setTimeout(() => {
          try {
            note.oscillator.stop();
            note.oscillator.disconnect();
            note.gain.disconnect();
          } catch (e) {
            // Already stopped
          }
        }, 60);
      }
      activeNotesRef.current.delete(pitch);
    }
  }, []);

  const stopAll = useCallback(() => {
    activeNotesRef.current.forEach((_, pitch) => {
      stopNote(pitch);
    });
  }, [stopNote]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAll();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [stopAll]);

  return { playNote, stopNote, stopAll };
};
