/**
 * Mixer Component
 * Main mixer view with all track channels and master channel
 * Connects to AudioEngine for real-time audio control
 *
 * Features modern glassmorphism design with Motion animations
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { useProjectStore } from '../../stores/projectStore';
import { getAudioEngine, initAudioEngine } from '../../audio';
import { Channel } from './Channel';
import { cn } from '../../lib/utils';

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
    <motion.div
      className={cn(
        'mixer',
        // Glassmorphism container
        'bg-slate-900/60 backdrop-blur-lg',
        'border border-slate-700/50',
        'rounded-xl shadow-xl',
        // Layout
        'flex flex-col',
        'overflow-hidden',
        'h-full',
        isMobile ? 'min-h-[220px]' : 'min-h-[280px]'
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      {/* Header */}
      <div
        className={cn(
          // Glass header background
          'bg-white/5 backdrop-blur-sm',
          'border-b border-white/10',
          // Layout
          'flex items-center justify-between shrink-0',
          isMobile ? 'px-3 py-2' : 'px-4 py-2.5'
        )}
      >
        <div className="flex items-center gap-2">
          {/* Title with gradient */}
          <span
            className={cn(
              'font-semibold bg-gradient-to-r from-rose-500 to-rose-400 bg-clip-text text-transparent',
              isMobile ? 'text-sm' : 'text-base'
            )}
          >
            Mixer
          </span>

          {/* Track count badge */}
          <span
            className={cn(
              'text-slate-400',
              'bg-slate-800/60',
              'px-2 py-0.5',
              'rounded-full',
              isMobile ? 'text-xs' : 'text-sm'
            )}
          >
            {project.tracks.length} tracks
          </span>
        </div>

        {/* Header actions could go here */}
      </div>

      {/* Channels */}
      <div
        className={cn(
          // Layout
          'flex flex-1 items-stretch',
          'overflow-x-auto',
          // Spacing
          isMobile ? 'gap-1.5 p-2.5' : 'gap-2 p-4'
        )}
        style={{
          WebkitOverflowScrolling: 'touch',
        }}
      >
        {/* Track channels */}
        {project.tracks.map((track, index) => (
          <motion.div
            key={track.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              duration: 0.2,
              delay: index * 0.03,
              ease: 'easeOut'
            }}
          >
            <Channel
              track={track}
              level={audioLevels[track.id] || 0}
            />
          </motion.div>
        ))}

        {/* Separator between tracks and master */}
        {project.tracks.length > 0 && (
          <div
            className={cn(
              'w-0.5 shrink-0',
              'bg-gradient-to-b from-rose-500/60 via-rose-500/30 to-transparent',
              'rounded-full',
              isMobile ? 'mx-1' : 'mx-2'
            )}
          />
        )}

        {/* Master channel */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            duration: 0.2,
            delay: project.tracks.length * 0.03,
            ease: 'easeOut'
          }}
        >
          <Channel
            track={project.masterTrack}
            level={masterLevel}
            isMaster={true}
          />
        </motion.div>
      </div>
    </motion.div>
  );
};

export default Mixer;
