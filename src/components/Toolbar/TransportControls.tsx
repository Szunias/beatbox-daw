/**
 * TransportControls Component
 * Play, stop, record buttons and position display
 */

import React, { useCallback } from 'react';
import { useTransportStore } from '../../stores/transportStore';
import { useProjectStore } from '../../stores/projectStore';
import { ticksToMeasures, TICKS_PER_BEAT } from '../../types/project';

export const TransportControls: React.FC = () => {
  const {
    state,
    currentTick,
    clickEnabled,
    loopRegion,
    play,
    pause,
    stop,
    record,
    toggleClick,
    setLoopEnabled,
  } = useTransportStore();

  const { project } = useProjectStore();
  const { timeSignatureNumerator, bpm } = project;

  const isPlaying = state === 'playing';
  const isRecording = state === 'recording';
  const isPaused = state === 'paused';
  const isStopped = state === 'stopped';

  // Format position as bars.beats.ticks
  const formatPosition = useCallback(
    (tick: number): string => {
      const ticksPerBar = TICKS_PER_BEAT * timeSignatureNumerator;
      const bars = Math.floor(tick / ticksPerBar) + 1;
      const beatsRemainder = tick % ticksPerBar;
      const beats = Math.floor(beatsRemainder / TICKS_PER_BEAT) + 1;
      const ticks = Math.floor(beatsRemainder % TICKS_PER_BEAT);
      return `${bars}.${beats}.${ticks.toString().padStart(3, '0')}`;
    },
    [timeSignatureNumerator]
  );

  const handlePlayPause = useCallback(() => {
    if (isPlaying || isRecording) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, isRecording, play, pause]);

  const handleStop = useCallback(() => {
    stop();
  }, [stop]);

  const handleRecord = useCallback(() => {
    record();
  }, [record]);

  return (
    <div
      className="transport-controls"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
      }}
    >
      {/* Transport buttons */}
      <div style={{ display: 'flex', gap: '4px' }}>
        {/* Stop button */}
        <button
          onClick={handleStop}
          title="Stop (Space)"
          style={{
            width: 36,
            height: 36,
            border: 'none',
            borderRadius: 4,
            backgroundColor: isStopped ? 'var(--text-secondary)' : 'rgba(255,255,255,0.1)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              width: 12,
              height: 12,
              backgroundColor: isStopped ? 'var(--bg-primary)' : 'var(--text-secondary)',
              borderRadius: 2,
            }}
          />
        </button>

        {/* Play/Pause button */}
        <button
          onClick={handlePlayPause}
          title="Play/Pause (Space)"
          style={{
            width: 36,
            height: 36,
            border: 'none',
            borderRadius: 4,
            backgroundColor: isPlaying ? 'var(--success)' : 'rgba(255,255,255,0.1)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {isPlaying || isRecording ? (
            // Pause icon
            <div style={{ display: 'flex', gap: 3 }}>
              <div
                style={{
                  width: 4,
                  height: 14,
                  backgroundColor: isPlaying ? 'var(--bg-primary)' : 'var(--text-secondary)',
                  borderRadius: 1,
                }}
              />
              <div
                style={{
                  width: 4,
                  height: 14,
                  backgroundColor: isPlaying ? 'var(--bg-primary)' : 'var(--text-secondary)',
                  borderRadius: 1,
                }}
              />
            </div>
          ) : (
            // Play icon
            <div
              style={{
                width: 0,
                height: 0,
                borderTop: '8px solid transparent',
                borderBottom: '8px solid transparent',
                borderLeft: '12px solid var(--text-secondary)',
                marginLeft: 3,
              }}
            />
          )}
        </button>

        {/* Record button */}
        <button
          onClick={handleRecord}
          title="Record (R)"
          style={{
            width: 36,
            height: 36,
            border: 'none',
            borderRadius: 4,
            backgroundColor: isRecording ? 'var(--error)' : 'rgba(255,255,255,0.1)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              width: 14,
              height: 14,
              backgroundColor: isRecording ? 'white' : 'var(--error)',
              borderRadius: '50%',
              animation: isRecording ? 'pulse 1s infinite' : 'none',
            }}
          />
        </button>
      </div>

      {/* Position display */}
      <div
        style={{
          fontFamily: 'monospace',
          fontSize: '1.1rem',
          fontWeight: 600,
          backgroundColor: 'var(--bg-primary)',
          padding: '6px 12px',
          borderRadius: 4,
          minWidth: 100,
          textAlign: 'center',
          color: isRecording ? 'var(--error)' : 'var(--text-primary)',
        }}
      >
        {formatPosition(currentTick)}
      </div>

      {/* Loop toggle */}
      <button
        onClick={() => setLoopEnabled(!loopRegion.enabled)}
        title="Toggle Loop (L) - Drag on timeline ruler to set loop region"
        style={{
          padding: '6px 10px',
          border: 'none',
          borderRadius: 4,
          backgroundColor: loopRegion.enabled ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
          color: loopRegion.enabled ? 'white' : 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '0.8rem',
          fontWeight: 600,
        }}
      >
        LOOP
      </button>

      {/* Metronome toggle */}
      <button
        onClick={toggleClick}
        title="Toggle Metronome"
        style={{
          padding: '6px 10px',
          border: 'none',
          borderRadius: 4,
          backgroundColor: clickEnabled ? 'var(--warning)' : 'rgba(255,255,255,0.1)',
          color: clickEnabled ? 'var(--bg-primary)' : 'var(--text-secondary)',
          cursor: 'pointer',
          fontSize: '0.8rem',
          fontWeight: 600,
        }}
      >
        CLICK
      </button>
    </div>
  );
};

export default TransportControls;
