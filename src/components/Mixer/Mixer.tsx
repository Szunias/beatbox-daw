/**
 * Mixer Component
 * Main mixer view with all track channels and master channel
 */

import React from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { Channel } from './Channel';

interface MixerProps {
  audioLevels?: Record<string, number>; // Track ID -> level (0-1)
  masterLevel?: number;
}

export const Mixer: React.FC<MixerProps> = ({ audioLevels = {}, masterLevel = 0 }) => {
  const { project } = useProjectStore();

  return (
    <div
      className="mixer"
      style={{
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '8px 12px',
          backgroundColor: 'var(--bg-tertiary)',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <span style={{ fontWeight: 500, fontSize: '0.85rem' }}>Mixer</span>
        <span
          style={{
            marginLeft: '8px',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
          }}
        >
          {project.tracks.length} tracks
        </span>
      </div>

      {/* Channels */}
      <div
        style={{
          display: 'flex',
          gap: '4px',
          padding: '12px',
          overflowX: 'auto',
          flex: 1,
        }}
      >
        {/* Track channels */}
        {project.tracks.map((track) => (
          <Channel
            key={track.id}
            track={track}
            level={audioLevels[track.id] || 0}
          />
        ))}

        {/* Separator */}
        {project.tracks.length > 0 && (
          <div
            style={{
              width: 1,
              backgroundColor: 'rgba(255,255,255,0.1)',
              margin: '0 8px',
            }}
          />
        )}

        {/* Master channel */}
        <Channel
          track={project.masterTrack}
          level={masterLevel}
          isMaster={true}
        />
      </div>
    </div>
  );
};

export default Mixer;
