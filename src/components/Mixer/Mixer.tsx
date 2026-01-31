/**
 * Mixer Component
 * Main mixer view with all track channels and master channel
 * Connects to AudioEngine for real-time audio control
 */

import React, { useState, useEffect } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { getAudioEngine, initAudioEngine } from '../../audio';
import { Channel } from './Channel';

interface MixerProps {
  audioLevels?: Record<string, number>; // Track ID -> level (0-1)
  masterLevel?: number;
}

export const Mixer: React.FC<MixerProps> = ({ audioLevels = {}, masterLevel = 0 }) => {
  const { project } = useProjectStore();
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize AudioEngine on mount
  useEffect(() => {
    const initEngine = async () => {
      await initAudioEngine();
    };
    initEngine();
  }, []);

  // Sync master track volume with AudioEngine
  useEffect(() => {
    const audioEngine = getAudioEngine();
    audioEngine.setMasterVolume(project.masterTrack.volume);
  }, [project.masterTrack.volume]);

  // Sync all track settings with AudioEngine when project changes
  useEffect(() => {
    const audioEngine = getAudioEngine();
    audioEngine.syncWithProject(project.tracks, project.bpm);
  }, [project.tracks, project.bpm]);

  return (
    <div
      className="mixer"
      style={{
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        overflow: 'hidden',
        height: '100%',
        minHeight: isMobile ? 220 : 280,
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: isMobile ? '8px 12px' : '10px 16px',
          backgroundColor: 'var(--bg-tertiary)',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontWeight: 600, fontSize: isMobile ? '0.9rem' : '1rem' }}>Mixer</span>
          <span
            style={{
              fontSize: isMobile ? '0.75rem' : '0.85rem',
              color: 'var(--text-secondary)',
              backgroundColor: 'var(--bg-primary)',
              padding: '2px 8px',
              borderRadius: 10,
            }}
          >
            {project.tracks.length} tracks
          </span>
        </div>
      </div>

      {/* Channels */}
      <div
        style={{
          display: 'flex',
          gap: isMobile ? '6px' : '8px',
          padding: isMobile ? '10px' : '16px',
          overflowX: 'auto',
          flex: 1,
          alignItems: 'stretch',
          WebkitOverflowScrolling: 'touch',
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
              width: 2,
              backgroundColor: 'var(--accent-primary)',
              margin: isMobile ? '0 4px' : '0 8px',
              borderRadius: 1,
              opacity: 0.5,
              flexShrink: 0,
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
