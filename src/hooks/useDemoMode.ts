/**
 * Demo Mode Hook
 * Provides simulated drum events and UI state when backend is unavailable
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { DrumEvent } from './useWebSocket';
import { demoPatterns, DemoPattern, generatePatternEvents } from '../data/demoPatterns';

interface UseDemoModeReturn {
  isDemoMode: boolean;
  setDemoMode: (enabled: boolean) => void;
  isPlaying: boolean;
  currentPattern: DemoPattern | null;
  demoEvents: DrumEvent[];
  simulatedAudioLevel: number;
  playPattern: (patternIndex?: number) => void;
  stopPattern: () => void;
  availablePatterns: DemoPattern[];
}

export function useDemoMode(): UseDemoModeReturn {
  const [isDemoMode, setDemoModeState] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentPattern, setCurrentPattern] = useState<DemoPattern | null>(null);
  const [demoEvents, setDemoEvents] = useState<DrumEvent[]>([]);
  const [simulatedAudioLevel, setSimulatedAudioLevel] = useState(0);

  const playbackRef = useRef<number | null>(null);
  const levelAnimationRef = useRef<number | null>(null);
  const eventIndexRef = useRef(0);

  const setDemoMode = useCallback((enabled: boolean) => {
    setDemoModeState(enabled);
    if (!enabled) {
      setIsPlaying(false);
      setCurrentPattern(null);
      setDemoEvents([]);
      setSimulatedAudioLevel(0);
    }
  }, []);

  const stopPattern = useCallback(() => {
    if (playbackRef.current) {
      clearInterval(playbackRef.current);
      playbackRef.current = null;
    }
    if (levelAnimationRef.current) {
      cancelAnimationFrame(levelAnimationRef.current);
      levelAnimationRef.current = null;
    }
    setIsPlaying(false);
    setSimulatedAudioLevel(0);
    eventIndexRef.current = 0;
  }, []);

  const playPattern = useCallback((patternIndex?: number) => {
    stopPattern();

    const pattern = patternIndex !== undefined
      ? demoPatterns[patternIndex]
      : demoPatterns[Math.floor(Math.random() * demoPatterns.length)];

    setCurrentPattern(pattern);
    setIsPlaying(true);
    setDemoEvents([]);
    eventIndexRef.current = 0;

    const events = generatePatternEvents(pattern);
    const startTime = Date.now();

    // Schedule events
    const playbackLoop = () => {
      const elapsed = (Date.now() - startTime) % pattern.duration;
      const loopStartTime = Date.now() - elapsed;

      // Find and emit events that should play now
      while (eventIndexRef.current < events.length) {
        const event = pattern.events[eventIndexRef.current];
        if (event.timestamp <= elapsed) {
          const newEvent: DrumEvent = {
            ...event,
            timestamp: loopStartTime + event.timestamp,
          };
          setDemoEvents((prev) => [newEvent, ...prev.slice(0, 99)]);

          // Simulate audio level spike
          const level = 0.5 + (event.velocity / 127) * 0.5;
          setSimulatedAudioLevel(level);
          setTimeout(() => {
            setSimulatedAudioLevel((prev) => Math.max(prev - 0.3, 0.1));
          }, 50);

          eventIndexRef.current++;
        } else {
          break;
        }
      }

      // Reset for loop
      if (elapsed < 50 && eventIndexRef.current >= events.length) {
        eventIndexRef.current = 0;
      }
    };

    playbackRef.current = window.setInterval(playbackLoop, 16);

    // Animate audio level decay
    const animateLevel = () => {
      setSimulatedAudioLevel((prev) => Math.max(prev * 0.95, 0.05));
      levelAnimationRef.current = requestAnimationFrame(animateLevel);
    };
    levelAnimationRef.current = requestAnimationFrame(animateLevel);
  }, [stopPattern]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
      if (levelAnimationRef.current) {
        cancelAnimationFrame(levelAnimationRef.current);
      }
    };
  }, []);

  return {
    isDemoMode,
    setDemoMode,
    isPlaying,
    currentPattern,
    demoEvents,
    simulatedAudioLevel,
    playPattern,
    stopPattern,
    availablePatterns: demoPatterns,
  };
}
