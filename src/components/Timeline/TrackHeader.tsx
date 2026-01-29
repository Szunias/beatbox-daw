/**
 * TrackHeader Component
 * Displays track name, mute/solo buttons, and volume/pan controls
 */

import React, { useCallback } from 'react';
import { Track } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';

interface TrackHeaderProps {
  track: Track;
  isSelected: boolean;
  height: number;
}

export const TrackHeader: React.FC<TrackHeaderProps> = ({ track, isSelected, height }) => {
  const {
    selectTrack,
    toggleTrackMute,
    toggleTrackSolo,
    toggleTrackArmed,
    setTrackVolume,
  } = useProjectStore();

  const handleClick = useCallback(() => {
    selectTrack(track.id);
  }, [track.id, selectTrack]);

  const handleMuteClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    toggleTrackMute(track.id);
  }, [track.id, toggleTrackMute]);

  const handleSoloClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    toggleTrackSolo(track.id);
  }, [track.id, toggleTrackSolo]);

  const handleArmClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    toggleTrackArmed(track.id);
  }, [track.id, toggleTrackArmed]);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    setTrackVolume(track.id, parseFloat(e.target.value));
  }, [track.id, setTrackVolume]);

  return (
    <div
      className={`track-header ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
      style={{
        height,
        display: 'flex',
        flexDirection: 'column',
        padding: '4px 8px',
        gap: '4px',
        backgroundColor: isSelected ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        cursor: 'pointer',
        transition: 'background-color 0.15s ease',
      }}
    >
      {/* Track name row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        {/* Color indicator */}
        <div
          style={{
            width: 4,
            height: 20,
            borderRadius: 2,
            backgroundColor: track.color,
          }}
        />

        {/* Track name */}
        <span
          style={{
            flex: 1,
            fontSize: '0.8rem',
            fontWeight: 500,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {track.name}
        </span>

        {/* Track type badge */}
        <span
          style={{
            fontSize: '0.65rem',
            padding: '1px 4px',
            borderRadius: 3,
            backgroundColor: 'rgba(255,255,255,0.1)',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
          }}
        >
          {track.type}
        </span>
      </div>

      {/* Controls row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        {/* Mute button */}
        <button
          className={`track-button ${track.muted ? 'active' : ''}`}
          onClick={handleMuteClick}
          title="Mute"
          style={{
            width: 22,
            height: 18,
            border: 'none',
            borderRadius: 3,
            fontSize: '0.65rem',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: track.muted ? 'var(--warning)' : 'rgba(255,255,255,0.1)',
            color: track.muted ? 'var(--bg-primary)' : 'var(--text-secondary)',
          }}
        >
          M
        </button>

        {/* Solo button */}
        <button
          className={`track-button ${track.solo ? 'active' : ''}`}
          onClick={handleSoloClick}
          title="Solo"
          style={{
            width: 22,
            height: 18,
            border: 'none',
            borderRadius: 3,
            fontSize: '0.65rem',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: track.solo ? 'var(--success)' : 'rgba(255,255,255,0.1)',
            color: track.solo ? 'var(--bg-primary)' : 'var(--text-secondary)',
          }}
        >
          S
        </button>

        {/* Arm button (record enable) */}
        <button
          className={`track-button ${track.armed ? 'active' : ''}`}
          onClick={handleArmClick}
          title="Record Arm"
          style={{
            width: 22,
            height: 18,
            border: 'none',
            borderRadius: 3,
            fontSize: '0.65rem',
            fontWeight: 600,
            cursor: 'pointer',
            backgroundColor: track.armed ? 'var(--error)' : 'rgba(255,255,255,0.1)',
            color: track.armed ? 'white' : 'var(--text-secondary)',
          }}
        >
          R
        </button>

        {/* Volume slider */}
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={track.volume}
          onChange={handleVolumeChange}
          onClick={(e) => e.stopPropagation()}
          title={`Volume: ${Math.round(track.volume * 100)}%`}
          style={{
            flex: 1,
            height: 4,
            cursor: 'pointer',
            accentColor: track.color,
          }}
        />

        {/* Volume value */}
        <span
          style={{
            fontSize: '0.65rem',
            color: 'var(--text-secondary)',
            minWidth: 28,
            textAlign: 'right',
          }}
        >
          {Math.round(track.volume * 100)}
        </span>
      </div>
    </div>
  );
};

export default TrackHeader;
