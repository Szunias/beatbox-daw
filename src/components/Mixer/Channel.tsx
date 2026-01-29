/**
 * Channel Component
 * Single mixer channel with fader, pan, mute, solo, and VU meter
 */

import React, { useCallback, useState } from 'react';
import { Track } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';
import { VUMeter } from './VUMeter';

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

  const isSelected = selectedTrackId === track.id;

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
        padding: '8px',
        gap: '8px',
        backgroundColor: isSelected ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
        borderRadius: 6,
        width: isMaster ? 80 : 70,
        cursor: 'pointer',
        border: isSelected ? '1px solid var(--accent-primary)' : '1px solid transparent',
        transition: 'background-color 0.15s ease, border-color 0.15s ease',
      }}
    >
      {/* Channel name */}
      <div
        style={{
          width: '100%',
          textAlign: 'center',
          fontSize: '0.7rem',
          fontWeight: 600,
          color: 'var(--text-primary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
        title={track.name}
      >
        {track.name}
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
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleMuteClick();
            }}
            style={{
              width: 24,
              height: 18,
              border: 'none',
              borderRadius: 3,
              fontSize: '0.6rem',
              fontWeight: 700,
              cursor: 'pointer',
              backgroundColor: track.muted ? 'var(--warning)' : 'rgba(255,255,255,0.1)',
              color: track.muted ? 'var(--bg-primary)' : 'var(--text-secondary)',
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
              width: 24,
              height: 18,
              border: 'none',
              borderRadius: 3,
              fontSize: '0.6rem',
              fontWeight: 700,
              cursor: 'pointer',
              backgroundColor: track.solo ? 'var(--success)' : 'rgba(255,255,255,0.1)',
              color: track.solo ? 'var(--bg-primary)' : 'var(--text-secondary)',
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
          gap: '2px',
          width: '100%',
        }}
      >
        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>PAN</span>
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
            height: 6,
            cursor: 'pointer',
            accentColor: track.color,
          }}
        />
        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>
          {formatPan(track.pan)}
        </span>
      </div>

      {/* VU Meter and Fader */}
      <div
        style={{
          display: 'flex',
          gap: '6px',
          alignItems: 'flex-end',
          flex: 1,
        }}
      >
        {/* VU Meter */}
        <VUMeter level={level * track.volume * (track.muted ? 0 : 1)} height={100} width={10} />

        {/* Volume fader */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            height: 100,
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
              width: 100,
              height: 20,
              cursor: 'pointer',
              accentColor: track.color,
            }}
          />
        </div>
      </div>

      {/* Volume dB display */}
      <div
        style={{
          backgroundColor: 'var(--bg-primary)',
          padding: '2px 6px',
          borderRadius: 3,
          fontSize: '0.65rem',
          fontFamily: 'monospace',
          color: 'var(--text-secondary)',
        }}
      >
        {formatVolumeDb(track.volume)}
      </div>
    </div>
  );
};

export default Channel;
