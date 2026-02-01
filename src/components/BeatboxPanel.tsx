/**
 * BeatboxPanel Component
 * Integration panel for beatbox recording and conversion to MIDI
 */

import React, { useState, useCallback } from 'react';
import { useWebSocket, DrumEvent } from '../hooks/useWebSocket';
import { useProjectStore } from '../stores/projectStore';
import { useTransportStore } from '../stores/transportStore';
import { WaveformVisualizer } from './WaveformVisualizer';

export const BeatboxPanel: React.FC = () => {
  const {
    isConnected,
    isDemoMode,
    status,
    audioLevel,
    recentEvents,
    startEngine,
    stopEngine,
    startRecording,
    stopRecording,
    exportMidi,
    playDemoPattern,
    stopDemoPattern,
    availablePatterns,
  } = useWebSocket();

  const { project, addBeatboxEvents, selectedTrackId } = useProjectStore();
  const { currentTick, state: transportState } = useTransportStore();

  const [isRecording, setIsRecording] = useState(false);
  const [recordedEvents, setRecordedEvents] = useState<DrumEvent[]>([]);
  const [recordStartTime, setRecordStartTime] = useState<number | null>(null);
  const [isDemoPlaying, setIsDemoPlaying] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState(0);

  // Find drum tracks
  const drumTracks = project.tracks.filter((t) => t.type === 'drum');
  const targetTrack = selectedTrackId
    ? project.tracks.find((t) => t.id === selectedTrackId)
    : drumTracks[0];

  const handleToggleRecording = useCallback(() => {
    if (isRecording) {
      // Stop recording
      stopEngine();
      setIsRecording(false);

      // Add recorded events to track
      if (recordedEvents.length > 0 && targetTrack) {
        // Normalize timestamps relative to recording start
        const normalizedEvents = recordedEvents.map((e) => ({
          ...e,
          timestamp: e.timestamp - (recordedEvents[0]?.timestamp || 0),
        }));
        addBeatboxEvents(targetTrack.id, normalizedEvents, currentTick);
      }

      setRecordedEvents([]);
      setRecordStartTime(null);
    } else {
      // Start recording
      startEngine();
      setIsRecording(true);
      setRecordedEvents([]);
      setRecordStartTime(Date.now());
    }
  }, [
    isRecording,
    recordedEvents,
    targetTrack,
    currentTick,
    startEngine,
    stopEngine,
    addBeatboxEvents,
  ]);

  // Demo mode toggle
  const handleToggleDemo = useCallback(() => {
    if (isDemoPlaying) {
      stopDemoPattern();
      setIsDemoPlaying(false);
    } else {
      playDemoPattern(selectedPattern);
      setIsDemoPlaying(true);
    }
  }, [isDemoPlaying, selectedPattern, playDemoPattern, stopDemoPattern]);

  // Collect events during recording
  React.useEffect(() => {
    if (isRecording && recentEvents.length > 0) {
      const latestEvent = recentEvents[0];
      // Check if this is a new event (not already collected)
      setRecordedEvents((prev) => {
        if (prev.some((e) => e.timestamp === latestEvent.timestamp)) {
          return prev;
        }
        return [...prev, latestEvent];
      });
    }
  }, [isRecording, recentEvents]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const drumColors: Record<string, string> = {
    kick: 'var(--accent-primary)',
    snare: 'var(--warning)',
    hihat: 'var(--success)',
    clap: '#60a5fa',
    tom: '#a78bfa',
  };

  return (
    <div
      className="beatbox-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        padding: '12px',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontWeight: 600 }}>BeatBox Input</span>
          {isDemoMode ? (
            <span
              style={{
                padding: '2px 8px',
                borderRadius: 4,
                backgroundColor: 'rgba(100, 100, 100, 0.2)',
                color: 'var(--text-secondary)',
                fontSize: '0.65rem',
                fontWeight: 500,
              }}
            >
              Offline
            </span>
          ) : (
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 8px',
                borderRadius: 4,
                backgroundColor: isConnected
                  ? 'rgba(34, 197, 94, 0.15)'
                  : 'rgba(239, 68, 68, 0.15)',
                fontSize: '0.65rem',
                fontWeight: 500,
                color: isConnected ? 'var(--success)' : 'var(--error)',
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  backgroundColor: isConnected ? 'var(--success)' : 'var(--error)',
                }}
              />
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          )}
        </div>
        {targetTrack && !isDemoMode && (
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Recording to: {targetTrack.name}
          </span>
        )}
      </div>

      {/* Waveform */}
      <div
        style={{
          height: 60,
          backgroundColor: 'var(--bg-primary)',
          borderRadius: 6,
          overflow: 'hidden',
        }}
      >
        <WaveformVisualizer audioLevel={audioLevel} isRecording={isRecording} />
      </div>

      {/* Recording controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        {isDemoMode ? (
          <>
            {/* Demo mode controls */}
            <select
              value={selectedPattern}
              onChange={(e) => setSelectedPattern(Number(e.target.value))}
              style={{
                padding: '10px 12px',
                border: 'none',
                borderRadius: 6,
                backgroundColor: 'var(--bg-primary)',
                color: 'var(--text-primary)',
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              {availablePatterns.map((pattern, idx) => (
                <option key={idx} value={idx}>
                  {pattern.name} ({pattern.bpm} BPM)
                </option>
              ))}
            </select>
            <button
              onClick={handleToggleDemo}
              style={{
                flex: 1,
                padding: '12px',
                border: 'none',
                borderRadius: 6,
                fontSize: '0.9rem',
                fontWeight: 600,
                cursor: 'pointer',
                backgroundColor: isDemoPlaying
                  ? 'var(--error)'
                  : 'var(--success)',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: isDemoPlaying ? 2 : '50%',
                  backgroundColor: 'white',
                }}
              />
              {isDemoPlaying ? 'Stop Demo' : 'Play Demo'}
            </button>
          </>
        ) : (
          <>
            {/* Normal recording controls */}
            <button
              onClick={handleToggleRecording}
              disabled={!isConnected || !targetTrack}
              style={{
                flex: 1,
                padding: '12px',
                border: 'none',
                borderRadius: 6,
                fontSize: '0.9rem',
                fontWeight: 600,
                cursor: isConnected && targetTrack ? 'pointer' : 'not-allowed',
                backgroundColor: isRecording
                  ? 'var(--error)'
                  : 'var(--accent-primary)',
                color: 'white',
                opacity: isConnected && targetTrack ? 1 : 0.5,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: isRecording ? 2 : '50%',
                  backgroundColor: 'white',
                }}
              />
              {isRecording ? 'Stop Recording' : 'Record BeatBox'}
            </button>
          </>
        )}

        {/* Stats */}
        <div
          style={{
            display: 'flex',
            gap: '16px',
            fontSize: '0.8rem',
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
              {isDemoMode ? recentEvents.length : recordedEvents.length}
            </div>
            <div style={{ color: 'var(--text-secondary)' }}>Events</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
              {Math.round(audioLevel * 100)}%
            </div>
            <div style={{ color: 'var(--text-secondary)' }}>Level</div>
          </div>
        </div>
      </div>

      {/* Recent events */}
      {recentEvents.length > 0 && (
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {recentEvents.slice(0, 16).map((event, idx) => (
            <div
              key={`${event.timestamp}-${idx}`}
              style={{
                padding: '2px 8px',
                borderRadius: 4,
                backgroundColor: drumColors[event.drum_class] || 'var(--bg-tertiary)',
                fontSize: '0.7rem',
                fontWeight: 500,
                color: 'white',
                textTransform: 'uppercase',
              }}
            >
              {event.drum_class}
            </div>
          ))}
        </div>
      )}

      {/* No drum track warning */}
      {!targetTrack && (
        <div
          style={{
            padding: '8px',
            backgroundColor: 'rgba(251, 191, 36, 0.1)',
            borderRadius: 4,
            fontSize: '0.8rem',
            color: 'var(--warning)',
            textAlign: 'center',
          }}
        >
          Add a drum track to record beatbox
        </div>
      )}

      {/* Connection warning */}
      {!isConnected && !isDemoMode && (
        <div
          style={{
            padding: '8px',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderRadius: 4,
            fontSize: '0.8rem',
            color: 'var(--error)',
            textAlign: 'center',
          }}
        >
          Engine disconnected. Start the Python engine to record.
        </div>
      )}

      {/* Offline mode notice */}
      {isDemoMode && (
        <div
          style={{
            padding: '6px 8px',
            backgroundColor: 'rgba(100, 100, 100, 0.1)',
            borderRadius: 4,
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            textAlign: 'center',
          }}
        >
          Backend offline. Start the Python engine for live beatbox detection.
        </div>
      )}
    </div>
  );
};

export default BeatboxPanel;
