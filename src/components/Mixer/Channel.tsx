/**
 * Channel Component
 * Single mixer channel with fader, pan, mute, solo, and VU meter
 * Connected to AudioEngine for real-time audio control
 */

import React, { useCallback, useState, useEffect } from 'react';
import { Track } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';
import { getAudioEngine } from '../../audio';
import { VUMeter } from './VUMeter';
import { EffectRack } from './EffectRack';

interface ChannelProps {
  track: Track;
  level?: number; // Audio level from engine (0-1)
  isMaster?: boolean;
}

export const Channel: React.FC<ChannelProps> = ({ track, level = 0, isMaster = false }) => {
  const {
    setTrackVolume,
    setTrackPan,
    toggleTrackMute,
    toggleTrackSolo,
    selectTrack,
    selectedTrackId,
  } = useProjectStore();

  const [isEditing, setIsEditing] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [showEffectRack, setShowEffectRack] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Sync track settings with AudioEngine for real-time audio control
  useEffect(() => {
    const audioEngine = getAudioEngine();
    audioEngine.updateTrackSettings(
      track.id,
      track.volume,
      track.pan,
      track.muted,
      track.solo
    );
  }, [track.id, track.volume, track.pan, track.muted, track.solo]);

  const isSelected = selectedTrackId === track.id;

  // Responsive sizing
  const channelWidth = isMobile ? 60 : (isMaster ? 90 : 80);
  const faderHeight = isMobile ? 80 : 120;
  const fontSize = {
    name: isMobile ? '0.7rem' : '0.8rem',
    label: isMobile ? '0.6rem' : '0.7rem',
    value: isMobile ? '0.65rem' : '0.75rem',
    button: isMobile ? '0.6rem' : '0.7rem',
  };

  const handleVolumeChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTrackVolume(track.id, parseFloat(e.target.value));
    },
    [track.id, setTrackVolume]
  );

  const handlePanChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setTrackPan(track.id, parseFloat(e.target.value));
    },
    [track.id, setTrackPan]
  );

  const handleMuteClick = useCallback(() => {
    toggleTrackMute(track.id);
  }, [track.id, toggleTrackMute]);

  const handleSoloClick = useCallback(() => {
    toggleTrackSolo(track.id);
  }, [track.id, toggleTrackSolo]);

  const handleChannelClick = useCallback(() => {
    selectTrack(track.id);
  }, [track.id, selectTrack]);

  // Format pan display
  const formatPan = (pan: number): string => {
    if (Math.abs(pan) < 0.01) return 'C';
    if (pan < 0) return `L${Math.abs(Math.round(pan * 100))}`;
    return `R${Math.round(pan * 100)}`;
  };

  // Format volume as dB
  const formatVolumeDb = (vol: number): string => {
    if (vol < 0.001) return '-∞';
    const db = 20 * Math.log10(vol);
    return `${db >= 0 ? '+' : ''}${db.toFixed(1)}`;
  };

  return (
    <div
      className={`mixer-channel ${isSelected ? 'selected' : ''} ${isMaster ? 'master' : ''}`}
      onClick={handleChannelClick}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: isMobile ? '6px' : '10px',
        gap: isMobile ? '6px' : '8px',
        backgroundColor: isSelected ? 'var(--bg-tertiary)' : 'var(--bg-primary)',
        borderRadius: 8,
        width: channelWidth,
        minWidth: channelWidth,
        cursor: 'pointer',
        border: isSelected ? '2px solid var(--accent-primary)' : '2px solid var(--bg-tertiary)',
        transition: 'background-color 0.15s ease, border-color 0.15s ease',
        flexShrink: 0,
        position: 'relative',
      }}
    >
      {/* Channel name */}
      <div
        style={{
          width: '100%',
          textAlign: 'center',
          fontSize: fontSize.name,
          fontWeight: 600,
          color: isMaster ? 'var(--accent-primary)' : 'var(--text-primary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          padding: '2px 0',
        }}
        title={track.name}
      >
        {isMaster ? 'MASTER' : track.name}
      </div>

      {/* Color indicator */}
      <div
        style={{
          width: '100%',
          height: 3,
          backgroundColor: track.color,
          borderRadius: 2,
        }}
      />

      {/* Mute/Solo buttons */}
      {!isMaster && (
        <div style={{ display: 'flex', gap: '4px', width: '100%', justifyContent: 'center' }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleMuteClick();
            }}
            style={{
              flex: 1,
              maxWidth: 32,
              height: isMobile ? 22 : 24,
              border: 'none',
              borderRadius: 4,
              fontSize: fontSize.button,
              fontWeight: 700,
              cursor: 'pointer',
              backgroundColor: track.muted ? 'var(--warning)' : 'var(--bg-tertiary)',
              color: track.muted ? 'var(--bg-primary)' : 'var(--text-secondary)',
              transition: 'background-color 0.1s ease',
            }}
          >
            M
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleSoloClick();
            }}
            style={{
              flex: 1,
              maxWidth: 32,
              height: isMobile ? 22 : 24,
              border: 'none',
              borderRadius: 4,
              fontSize: fontSize.button,
              fontWeight: 700,
              cursor: 'pointer',
              backgroundColor: track.solo ? 'var(--success)' : 'var(--bg-tertiary)',
              color: track.solo ? 'var(--bg-primary)' : 'var(--text-secondary)',
              transition: 'background-color 0.1s ease',
            }}
          >
            S
          </button>
        </div>
      )}

      {/* Pan control */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '3px',
          width: '100%',
          padding: '4px 0',
        }}
      >
        <span style={{ fontSize: fontSize.label, color: 'var(--text-secondary)', fontWeight: 500 }}>PAN</span>
        <input
          type="range"
          min="-1"
          max="1"
          step="0.01"
          value={track.pan}
          onChange={handlePanChange}
          onClick={(e) => e.stopPropagation()}
          onDoubleClick={(e) => {
            e.stopPropagation();
            setTrackPan(track.id, 0); // Reset to center
          }}
          style={{
            width: '100%',
            height: isMobile ? 20 : 8,
            cursor: 'pointer',
            accentColor: isMaster ? 'var(--accent-primary)' : track.color,
          }}
        />
        <span style={{ fontSize: fontSize.value, color: 'var(--text-primary)', fontFamily: 'monospace' }}>
          {formatPan(track.pan)}
        </span>
      </div>

      {/* VU Meter and Fader */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'stretch',
          flex: 1,
          width: '100%',
          justifyContent: 'center',
          minHeight: faderHeight,
        }}
      >
        {/* VU Meter */}
        <VUMeter
          level={level * track.volume * (track.muted ? 0 : 1)}
          height={faderHeight}
          width={isMobile ? 8 : 12}
        />

        {/* Volume fader */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: faderHeight,
          }}
        >
          <input
            type="range"
            min="0"
            max="1.5"
            step="0.01"
            value={track.volume}
            onChange={handleVolumeChange}
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => {
              e.stopPropagation();
              setTrackVolume(track.id, 0.8); // Reset to default
            }}
            style={{
              writingMode: 'vertical-lr',
              direction: 'rtl',
              width: faderHeight,
              height: isMobile ? 24 : 28,
              cursor: 'pointer',
              accentColor: isMaster ? 'var(--accent-primary)' : track.color,
            }}
          />
        </div>
      </div>

      {/* Volume dB display */}
      <div
        style={{
          backgroundColor: 'var(--bg-tertiary)',
          padding: isMobile ? '3px 6px' : '4px 8px',
          borderRadius: 4,
          fontSize: fontSize.value,
          fontFamily: 'monospace',
          color: 'var(--text-primary)',
          fontWeight: 500,
          minWidth: isMobile ? 40 : 48,
          textAlign: 'center',
        }}
      >
        {formatVolumeDb(track.volume)}
      </div>

      {/* FX Button - opens Effect Rack */}
      {!isMaster && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowEffectRack(!showEffectRack);
          }}
          style={{
            width: '100%',
            height: isMobile ? 22 : 24,
            border: 'none',
            borderRadius: 4,
            fontSize: fontSize.button,
            fontWeight: 700,
            cursor: 'pointer',
            backgroundColor: showEffectRack ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
            color: showEffectRack ? 'var(--bg-primary)' : 'var(--text-secondary)',
            transition: 'background-color 0.1s ease',
          }}
          title="Toggle Effect Rack"
        >
          FX
        </button>
      )}

      {/* Effect Rack - shown when FX button is clicked */}
      {!isMaster && showEffectRack && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: 4,
            zIndex: 100,
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          }}
        >
          <EffectRack
            trackId={track.id}
            trackColor={track.color}
            maxEffects={4}
          />
        </div>
      )}
    </div>
  );
};

export default Channel;
