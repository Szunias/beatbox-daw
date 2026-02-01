/**
 * Channel Component
 * Single mixer channel with fader, pan, mute, solo, and VU meter
 * Connected to AudioEngine for real-time audio control
 *
 * Features modern glassmorphism design with Motion animations
 */

import React, { useCallback, useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Track } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';
import { getAudioEngine } from '../../audio';
import { VUMeter } from './VUMeter';
import { EffectRack } from './EffectRack';
import { cn } from '../../lib/utils';

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
    <motion.div
      className={cn(
        'mixer-channel',
        // Glassmorphism effect
        'bg-slate-800/50 backdrop-blur-md',
        'border rounded-lg',
        isSelected
          ? 'border-rose-500/60 shadow-lg shadow-rose-500/10'
          : 'border-slate-700/50',
        isMaster && 'bg-slate-800/70',
        // Layout
        'flex flex-col items-center',
        'cursor-pointer',
        'relative',
        'shrink-0',
        // Spacing
        isMobile ? 'p-1.5 gap-1.5' : 'p-2.5 gap-2',
        // Transitions
        'transition-all duration-200 ease-out'
      )}
      style={{
        width: channelWidth,
        minWidth: channelWidth,
      }}
      onClick={handleChannelClick}
      whileHover={{
        scale: 1.02,
        borderColor: 'rgba(244, 63, 94, 0.4)'
      }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
    >
      {/* Channel name */}
      <div
        className={cn(
          'w-full text-center font-semibold',
          'overflow-hidden text-ellipsis whitespace-nowrap',
          'py-0.5',
          isMaster
            ? 'text-rose-400'
            : 'text-slate-100',
          isMobile ? 'text-[0.7rem]' : 'text-[0.8rem]'
        )}
        title={track.name}
      >
        {isMaster ? 'MASTER' : track.name}
      </div>

      {/* Color indicator with glow */}
      <div
        className="w-full h-0.5 rounded-full"
        style={{
          backgroundColor: track.color,
          boxShadow: `0 0 8px ${track.color}40`,
        }}
      />

      {/* Mute/Solo buttons */}
      {!isMaster && (
        <div className="flex gap-1 w-full justify-center">
          {/* Mute button */}
          <motion.button
            onClick={(e) => {
              e.stopPropagation();
              handleMuteClick();
            }}
            className={cn(
              'flex-1 max-w-8 font-bold rounded',
              'border border-transparent',
              'transition-colors duration-150',
              track.muted
                ? 'bg-amber-500 text-slate-900 border-amber-400'
                : 'bg-slate-700/60 text-slate-400 hover:bg-slate-700 hover:text-slate-300',
              isMobile ? 'h-[22px] text-[0.6rem]' : 'h-6 text-[0.7rem]'
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 500, damping: 25 }}
          >
            M
          </motion.button>

          {/* Solo button */}
          <motion.button
            onClick={(e) => {
              e.stopPropagation();
              handleSoloClick();
            }}
            className={cn(
              'flex-1 max-w-8 font-bold rounded',
              'border border-transparent',
              'transition-colors duration-150',
              track.solo
                ? 'bg-green-500 text-slate-900 border-green-400'
                : 'bg-slate-700/60 text-slate-400 hover:bg-slate-700 hover:text-slate-300',
              isMobile ? 'h-[22px] text-[0.6rem]' : 'h-6 text-[0.7rem]'
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 500, damping: 25 }}
          >
            S
          </motion.button>
        </div>
      )}

      {/* Pan control */}
      <div className="flex flex-col items-center gap-0.5 w-full py-1">
        <span
          className={cn(
            'text-slate-500 font-medium',
            isMobile ? 'text-[0.6rem]' : 'text-[0.7rem]'
          )}
        >
          PAN
        </span>
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
            setTrackPan(track.id, 0);
          }}
          className={cn(
            'w-full cursor-pointer',
            'appearance-none bg-transparent',
            '[&::-webkit-slider-runnable-track]:h-1',
            '[&::-webkit-slider-runnable-track]:rounded-full',
            '[&::-webkit-slider-runnable-track]:bg-slate-700',
            '[&::-webkit-slider-thumb]:appearance-none',
            '[&::-webkit-slider-thumb]:w-3',
            '[&::-webkit-slider-thumb]:h-3',
            '[&::-webkit-slider-thumb]:rounded-full',
            '[&::-webkit-slider-thumb]:bg-slate-300',
            '[&::-webkit-slider-thumb]:mt-[-4px]',
            '[&::-webkit-slider-thumb]:transition-all',
            '[&::-webkit-slider-thumb]:duration-150',
            '[&::-webkit-slider-thumb]:hover:bg-white',
            '[&::-webkit-slider-thumb]:hover:scale-125',
            isMobile ? 'h-5' : 'h-4'
          )}
          style={{
            accentColor: isMaster ? 'var(--accent-primary)' : track.color,
          }}
        />
        <span
          className={cn(
            'text-slate-200 font-mono',
            isMobile ? 'text-[0.65rem]' : 'text-[0.75rem]'
          )}
        >
          {formatPan(track.pan)}
        </span>
      </div>

      {/* VU Meter and Fader */}
      <div
        className="flex gap-2 items-stretch flex-1 w-full justify-center"
        style={{ minHeight: faderHeight }}
      >
        {/* VU Meter */}
        <VUMeter
          level={level * track.volume * (track.muted ? 0 : 1)}
          height={faderHeight}
          width={isMobile ? 8 : 12}
        />

        {/* Volume fader */}
        <div
          className="flex flex-col items-center justify-center"
          style={{ height: faderHeight }}
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
              setTrackVolume(track.id, 0.8);
            }}
            className={cn(
              'cursor-pointer',
              'appearance-none bg-transparent',
              '[&::-webkit-slider-runnable-track]:w-1',
              '[&::-webkit-slider-runnable-track]:rounded-full',
              '[&::-webkit-slider-runnable-track]:bg-gradient-to-t',
              '[&::-webkit-slider-runnable-track]:from-slate-700',
              '[&::-webkit-slider-runnable-track]:to-slate-600',
              '[&::-webkit-slider-thumb]:appearance-none',
              '[&::-webkit-slider-thumb]:w-6',
              '[&::-webkit-slider-thumb]:h-2.5',
              '[&::-webkit-slider-thumb]:rounded-sm',
              '[&::-webkit-slider-thumb]:bg-gradient-to-b',
              '[&::-webkit-slider-thumb]:from-slate-200',
              '[&::-webkit-slider-thumb]:to-slate-400',
              '[&::-webkit-slider-thumb]:shadow-md',
              '[&::-webkit-slider-thumb]:transition-all',
              '[&::-webkit-slider-thumb]:duration-100',
              '[&::-webkit-slider-thumb]:hover:from-white',
              '[&::-webkit-slider-thumb]:hover:to-slate-300',
              '[&::-webkit-slider-thumb]:active:from-rose-200',
              '[&::-webkit-slider-thumb]:active:to-rose-400'
            )}
            style={{
              writingMode: 'vertical-lr',
              direction: 'rtl',
              width: faderHeight,
              height: isMobile ? 24 : 28,
              accentColor: isMaster ? 'var(--accent-primary)' : track.color,
            }}
          />
        </div>
      </div>

      {/* Volume dB display */}
      <div
        className={cn(
          'bg-slate-900/70 backdrop-blur-sm',
          'border border-slate-700/50',
          'rounded font-mono font-medium',
          'text-center text-slate-100',
          isMobile
            ? 'px-1.5 py-0.5 text-[0.65rem] min-w-10'
            : 'px-2 py-1 text-[0.75rem] min-w-12'
        )}
      >
        {formatVolumeDb(track.volume)}
      </div>

      {/* FX Button - opens Effect Rack */}
      {!isMaster && (
        <motion.button
          onClick={(e) => {
            e.stopPropagation();
            setShowEffectRack(!showEffectRack);
          }}
          className={cn(
            'w-full font-bold rounded',
            'border border-transparent',
            'transition-colors duration-150',
            showEffectRack
              ? 'bg-rose-500 text-slate-900 border-rose-400'
              : 'bg-slate-700/60 text-slate-400 hover:bg-slate-700 hover:text-slate-300',
            isMobile ? 'h-[22px] text-[0.6rem]' : 'h-6 text-[0.7rem]'
          )}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: 'spring', stiffness: 500, damping: 25 }}
          title="Toggle Effect Rack"
        >
          FX
        </motion.button>
      )}

      {/* Effect Rack - shown when FX button is clicked */}
      {!isMaster && showEffectRack && (
        <motion.div
          onClick={(e) => e.stopPropagation()}
          className={cn(
            'absolute top-full left-0 mt-1 z-[100]',
            'shadow-xl shadow-black/40'
          )}
          initial={{ opacity: 0, y: -8, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.95 }}
          transition={{ duration: 0.15, ease: 'easeOut' }}
        >
          <EffectRack
            trackId={track.id}
            trackColor={track.color}
            maxEffects={4}
          />
        </motion.div>
      )}
    </motion.div>
  );
};

export default Channel;
