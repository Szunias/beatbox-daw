/**
 * BeatBox DAW - Main Application
 * Full Digital Audio Workstation with BeatBox-to-MIDI conversion
 */

import React, { useEffect, useState, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useProjectStore } from './stores/projectStore';
import { useTransportStore } from './stores/transportStore';
import { useUIStore } from './stores/uiStore';
import { Toolbar } from './components/Toolbar';
import { Timeline } from './components/Timeline';
import { PianoRoll } from './components/PianoRoll';
import { Mixer } from './components/Mixer';
import { BeatboxPanel } from './components/BeatboxPanel';
import { AudioSettings } from './components/Settings';
import { AccessGate } from './components/AccessGate';
import { TICKS_PER_BEAT } from './types/project';

const App: React.FC = () => {
  const {
    isConnected,
    isDemoMode,
    audioLevel,
    sendMessage,
    onTransportPosition,
    onRecordingCompleted,
    devices,
    listDevices,
    setAudioDevice,
  } = useWebSocket();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const { project, setWebSocket: setProjectWebSocket, addAudioClipFromRecording } = useProjectStore();
  const {
    state: transportState,
    isRecording,
    currentTick,
    setCurrentTick,
    play,
    pause,
    stop,
    record,
    setWebSocket,
    syncFromBackend,
  } = useTransportStore();
  const {
    isPianoRollOpen,
    isMixerOpen,
    isBeatboxPanelOpen,
    toggleMixer,
    toggleBeatboxPanel,
    setIsPlaying,
  } = useUIStore();

  // Connect transport store to WebSocket
  useEffect(() => {
    setWebSocket(sendMessage, isConnected);
  }, [sendMessage, isConnected, setWebSocket]);

  // Connect project store to WebSocket (for BPM sync)
  useEffect(() => {
    setProjectWebSocket(sendMessage, isConnected);
  }, [sendMessage, isConnected, setProjectWebSocket]);

  // Subscribe to transport position updates from backend
  useEffect(() => {
    const unsubscribe = onTransportPosition((position) => {
      // Sync playhead position from backend
      const backendState = position.state as 'stopped' | 'playing' | 'paused' | 'recording';
      syncFromBackend(position.tick, backendState);
    });
    return unsubscribe;
  }, [onTransportPosition, syncFromBackend]);

  // Subscribe to recording completed events
  useEffect(() => {
    const unsubscribe = onRecordingCompleted((data) => {
      // Create audio clip from recording on the specified track
      if (data.trackId && data.durationSeconds > 0) {
        // Use a placeholder audio file path - the backend stores the recording
        // and the path will be determined by the project's audio file management
        const audioFilePath = data.audioFilePath || `recordings/${data.trackId}_${Date.now()}.wav`;
        addAudioClipFromRecording(
          data.trackId,
          audioFilePath,
          data.startTick,
          data.durationSeconds
        );
      }
    });
    return unsubscribe;
  }, [onRecordingCompleted, addAudioClipFromRecording]);

  // Update UI store when transport state changes
  useEffect(() => {
    setIsPlaying(transportState === 'playing' || transportState === 'recording');
  }, [transportState, setIsPlaying]);

  // Use ref to track current tick without stale closure issues
  const tickRef = useRef(currentTick);
  useEffect(() => {
    tickRef.current = currentTick;
  }, [currentTick]);

  // Playback position update loop - only used when NOT connected to backend
  // When connected, backend sends position updates via WebSocket
  useEffect(() => {
    // Skip local playback loop if connected to backend - backend handles position
    if (isConnected) return;
    if (transportState !== 'playing' && transportState !== 'recording') return;

    let animationFrameId: number;
    let lastTime = performance.now();

    const updatePosition = (currentTime: number) => {
      const deltaMs = currentTime - lastTime;
      const deltaTicks = (deltaMs / 1000) * (project.bpm / 60) * TICKS_PER_BEAT;

      const newTick = tickRef.current + deltaTicks;
      tickRef.current = newTick;
      setCurrentTick(newTick);
      lastTime = currentTime;

      animationFrameId = requestAnimationFrame(updatePosition);
    };

    animationFrameId = requestAnimationFrame(updatePosition);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [transportState, project.bpm, setCurrentTick, isConnected]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          if (transportState === 'playing' || transportState === 'recording') {
            pause();
          } else {
            play();
          }
          break;
        case 'Enter':
          e.preventDefault();
          stop();
          break;
        case 'KeyR':
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            record();
          }
          break;
        case 'KeyM':
          if (!e.ctrlKey && !e.metaKey) {
            toggleMixer();
          }
          break;
        case 'KeyB':
          if (!e.ctrlKey && !e.metaKey) {
            toggleBeatboxPanel();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [transportState, play, pause, stop, record, toggleMixer, toggleBeatboxPanel]);

  // Calculate dynamic heights
  const headerHeight = 56; // Toolbar
  const bottomPanelHeight = isMixerOpen || isBeatboxPanelOpen ? 220 : 0;
  const pianoRollHeight = isPianoRollOpen ? 280 : 0;
  const timelineHeight = `calc(100vh - ${headerHeight + bottomPanelHeight + pianoRollHeight + 40}px)`;

  return (
    <AccessGate>
      <div className="app daw-layout">
        {/* Header with connection status */}
        <header className="header">
          <h1>BeatBox DAW</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {isRecording && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '4px 10px',
                  borderRadius: 4,
                  backgroundColor: 'rgba(239, 68, 68, 0.2)',
                  color: '#ef4444',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  animation: 'pulse 1s infinite',
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: '#ef4444',
                    animation: 'blink 1s infinite',
                  }}
                />
                REC
              </div>
            )}
            {isDemoMode && (
              <div
                style={{
                  padding: '4px 10px',
                  borderRadius: 4,
                  backgroundColor: 'rgba(251, 191, 36, 0.2)',
                  color: 'var(--warning)',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  border: '1px solid rgba(251, 191, 36, 0.3)',
                }}
              >
                DEMO MODE
              </div>
            )}
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              style={{
                padding: '6px 12px',
                border: 'none',
                borderRadius: 4,
                backgroundColor: isSettingsOpen ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
                color: isSettingsOpen ? 'white' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '0.8rem',
              }}
            >
              Settings
            </button>
            <div className="status-indicator">
              <span className={`status-dot ${isConnected ? 'connected' : ''}`} />
              {isDemoMode ? 'Demo Mode' : isConnected ? 'Engine Connected' : 'Connecting...'}
            </div>
          </div>
        </header>

      {/* Audio Settings Modal */}
      {isSettingsOpen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
          }}
          onClick={() => setIsSettingsOpen(false)}
        >
          <div onClick={(e) => e.stopPropagation()}>
            <AudioSettings
              onClose={() => setIsSettingsOpen(false)}
              isConnected={isConnected}
              devices={devices}
              listDevices={listDevices}
              setAudioDevice={setAudioDevice}
            />
          </div>
        </div>
      )}

      {/* Main DAW Content */}
      <main className="daw-main">
        {/* Toolbar */}
        <Toolbar />

        {/* Timeline */}
        <div style={{ height: timelineHeight, minHeight: 200 }}>
          <Timeline height={parseInt(timelineHeight) || 300} />
        </div>

        {/* Piano Roll (conditional) */}
        {isPianoRollOpen && (
          <PianoRoll height={pianoRollHeight} />
        )}

        {/* Bottom Panel: Mixer and BeatBox */}
        {(isMixerOpen || isBeatboxPanelOpen) && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: isMixerOpen && isBeatboxPanelOpen
                ? '1fr 300px'
                : '1fr',
              gap: '12px',
              height: bottomPanelHeight,
            }}
          >
            {isMixerOpen && (
              <Mixer
                audioLevels={{}}
                masterLevel={audioLevel}
              />
            )}
            {isBeatboxPanelOpen && (
              <BeatboxPanel />
            )}
          </div>
        )}

        {/* Panel toggles */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '8px',
            padding: '4px',
          }}
        >
          <button
            onClick={toggleMixer}
            style={{
              padding: '4px 12px',
              border: 'none',
              borderRadius: 4,
              backgroundColor: isMixerOpen ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
              color: isMixerOpen ? 'white' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '0.75rem',
            }}
          >
            Mixer (M)
          </button>
          <button
            onClick={toggleBeatboxPanel}
            style={{
              padding: '4px 12px',
              border: 'none',
              borderRadius: 4,
              backgroundColor: isBeatboxPanelOpen ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
              color: isBeatboxPanelOpen ? 'white' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '0.75rem',
            }}
          >
            BeatBox (B)
          </button>
        </div>
      </main>
    </div>
    </AccessGate>
  );
};

export default App;
