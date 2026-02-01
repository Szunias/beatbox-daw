/**
 * EffectRack Component
 * Manages effect chain for a track with add/remove/reorder functionality
 * Wires to AudioEngine's EffectsProcessor for real-time audio processing
 *
 * Features modern glassmorphism design with Motion animations
 */

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { getAudioEngine } from '../../audio';
import type { EffectType, EffectParameters } from '../../audio';
import { EffectEditor } from './EffectEditor';
import { cn } from '../../lib/utils';

interface EffectInfo {
  id: string;
  type: EffectType;
  enabled: boolean;
  parameters: EffectParameters;
}

interface EffectRackProps {
  trackId: string;
  trackColor?: string;
  maxEffects?: number;
}

// Available effect types
const AVAILABLE_EFFECTS: Array<{ type: EffectType; label: string }> = [
  { type: 'eq3band', label: 'EQ' },
  { type: 'compressor', label: 'Comp' },
  { type: 'delay', label: 'Delay' },
  { type: 'reverb', label: 'Reverb' },
];

export const EffectRack: React.FC<EffectRackProps> = ({
  trackId,
  trackColor = 'var(--accent-primary)',
  maxEffects = 8,
}) => {
  const [effects, setEffects] = useState<EffectInfo[]>([]);
  const [selectedEffectId, setSelectedEffectId] = useState<string | null>(null);
  const [bypassed, setBypassed] = useState(false);
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Sync effect chain state from AudioEngine
  const syncEffects = useCallback(() => {
    const audioEngine = getAudioEngine();
    const chainInfo = audioEngine.getTrackEffectsInfo(trackId);
    setEffects(chainInfo);
    setBypassed(audioEngine.isTrackEffectsBypassed(trackId));
  }, [trackId]);

  // Initial sync and periodic refresh
  useEffect(() => {
    syncEffects();

    // Refresh periodically to catch any external changes
    const interval = setInterval(syncEffects, 1000);
    return () => clearInterval(interval);
  }, [syncEffects]);

  // Add effect to chain
  const handleAddEffect = useCallback(
    (effectType: EffectType) => {
      if (effects.length >= maxEffects) return;

      const audioEngine = getAudioEngine();
      const effectId = audioEngine.addTrackEffect(trackId, effectType);

      if (effectId) {
        syncEffects();
        setSelectedEffectId(effectId);
      }
      setShowAddMenu(false);
    },
    [trackId, effects.length, maxEffects, syncEffects]
  );

  // Remove effect from chain
  const handleRemoveEffect = useCallback(
    (effectId: string) => {
      const audioEngine = getAudioEngine();
      const success = audioEngine.removeTrackEffect(trackId, effectId);

      if (success) {
        if (selectedEffectId === effectId) {
          setSelectedEffectId(null);
        }
        syncEffects();
      }
    },
    [trackId, selectedEffectId, syncEffects]
  );

  // Toggle effect enabled/disabled
  const handleToggleEffect = useCallback(
    (effectId: string, enabled: boolean) => {
      const audioEngine = getAudioEngine();
      audioEngine.setTrackEffectEnabled(trackId, effectId, enabled);
      syncEffects();
    },
    [trackId, syncEffects]
  );

  // Toggle bypass all effects
  const handleToggleBypass = useCallback(() => {
    const audioEngine = getAudioEngine();
    const newBypassed = !bypassed;
    audioEngine.setTrackEffectsBypass(trackId, newBypassed);
    setBypassed(newBypassed);
  }, [trackId, bypassed]);

  // Move effect in chain
  const handleMoveEffect = useCallback(
    (effectId: string, direction: 'up' | 'down') => {
      const index = effects.findIndex((e) => e.id === effectId);
      if (index === -1) return;

      const newPosition = direction === 'up' ? index - 1 : index + 1;
      if (newPosition < 0 || newPosition >= effects.length) return;

      const audioEngine = getAudioEngine();
      const success = audioEngine.moveTrackEffect(trackId, effectId, newPosition);

      if (success) {
        syncEffects();
      }
    },
    [trackId, effects, syncEffects]
  );

  // Handle parameter change
  const handleParameterChange = useCallback(() => {
    syncEffects();
  }, [syncEffects]);

  const selectedEffect = selectedEffectId ? effects.find((e) => e.id === selectedEffectId) : null;

  return (
    <motion.div
      className={cn(
        'effect-rack',
        'flex flex-col',
        // Glassmorphism effect
        'bg-slate-800/60 backdrop-blur-lg',
        'border border-slate-700/50',
        'rounded-xl shadow-xl',
        'overflow-hidden',
        isMobile ? 'min-w-[200px]' : 'min-w-[280px]'
      )}
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between',
          'bg-slate-900/60 backdrop-blur-md',
          'border-b border-slate-700/50',
          isMobile ? 'px-2 py-1.5' : 'px-3 py-2'
        )}
      >
        <div className="flex items-center gap-2">
          {/* Track color indicator */}
          <span
            className={cn(
              'w-1 rounded-full',
              isMobile ? 'h-3.5' : 'h-4'
            )}
            style={{ backgroundColor: trackColor }}
          />
          <span
            className={cn(
              'font-semibold text-slate-100',
              isMobile ? 'text-[0.8rem]' : 'text-[0.9rem]'
            )}
          >
            Effects
          </span>
          {/* Effect count badge */}
          <span
            className={cn(
              'bg-slate-700/60 text-slate-400',
              'px-1.5 py-0.5 rounded-md',
              'font-medium',
              isMobile ? 'text-[0.65rem]' : 'text-[0.75rem]'
            )}
          >
            {effects.length}/{maxEffects}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {/* Bypass button */}
          <motion.button
            onClick={handleToggleBypass}
            className={cn(
              'font-bold rounded',
              'border border-transparent',
              'transition-colors duration-150',
              bypassed
                ? 'bg-amber-500 text-slate-900 border-amber-400'
                : 'bg-slate-700/60 text-slate-400 hover:bg-slate-700 hover:text-slate-300',
              isMobile ? 'px-1.5 py-0.5 text-[0.6rem]' : 'px-2 py-1 text-[0.7rem]'
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 500, damping: 25 }}
            title={bypassed ? 'Enable effects' : 'Bypass all effects'}
          >
            BYP
          </motion.button>

          {/* Add button */}
          <div className="relative">
            <motion.button
              onClick={() => setShowAddMenu(!showAddMenu)}
              disabled={effects.length >= maxEffects}
              className={cn(
                'font-bold rounded',
                'transition-colors duration-150',
                effects.length >= maxEffects
                  ? 'bg-slate-700/40 text-slate-600 cursor-not-allowed'
                  : 'bg-rose-500 text-slate-100 hover:bg-rose-400',
                isMobile ? 'px-2 py-0.5 text-[0.7rem]' : 'px-2.5 py-1 text-[0.8rem]'
              )}
              whileHover={effects.length < maxEffects ? { scale: 1.05 } : undefined}
              whileTap={effects.length < maxEffects ? { scale: 0.92 } : undefined}
              transition={{ type: 'spring', stiffness: 500, damping: 25 }}
              title="Add effect"
            >
              +
            </motion.button>

            {/* Add effect menu */}
            <AnimatePresence>
              {showAddMenu && (
                <motion.div
                  className={cn(
                    'absolute top-full right-0 mt-1',
                    'bg-slate-800/90 backdrop-blur-lg',
                    'border border-slate-600/50',
                    'rounded-lg shadow-xl shadow-black/40',
                    'overflow-hidden',
                    'z-[100] min-w-[100px]'
                  )}
                  initial={{ opacity: 0, y: -8, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -8, scale: 0.95 }}
                  transition={{ duration: 0.15, ease: 'easeOut' }}
                >
                  {AVAILABLE_EFFECTS.map((effect) => (
                    <motion.button
                      key={effect.type}
                      onClick={() => handleAddEffect(effect.type)}
                      className={cn(
                        'block w-full text-left',
                        'text-slate-200',
                        'transition-colors duration-100',
                        'hover:bg-slate-700/80 hover:text-white',
                        isMobile ? 'px-2.5 py-1.5 text-[0.75rem]' : 'px-3 py-2 text-[0.85rem]'
                      )}
                      whileHover={{ x: 2 }}
                      transition={{ duration: 0.1 }}
                    >
                      {effect.label}
                    </motion.button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Effect slots */}
      <div
        className={cn(
          'flex flex-col gap-0.5',
          'min-h-[60px]',
          'transition-opacity duration-200',
          bypassed && 'opacity-50',
          isMobile ? 'p-1' : 'p-1.5'
        )}
      >
        {effects.length === 0 ? (
          <motion.div
            className={cn(
              'text-center text-slate-500',
              isMobile ? 'py-3 text-[0.75rem]' : 'py-4 text-[0.85rem]'
            )}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            No effects - click + to add
          </motion.div>
        ) : (
          effects.map((effect, index) => (
            <motion.div
              key={effect.id}
              onClick={() => setSelectedEffectId(effect.id === selectedEffectId ? null : effect.id)}
              className={cn(
                'flex items-center gap-1.5',
                'rounded-md cursor-pointer',
                'transition-all duration-150',
                effect.id === selectedEffectId
                  ? 'bg-slate-700/70 border border-rose-500/50'
                  : 'bg-slate-700/40 border border-transparent hover:bg-slate-700/60',
                isMobile ? 'px-1.5 py-1' : 'px-2 py-1.5'
              )}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              whileHover={{ scale: 1.01 }}
            >
              {/* Effect number */}
              <span
                className={cn(
                  'text-slate-500 text-center w-3.5',
                  isMobile ? 'text-[0.6rem]' : 'text-[0.65rem]'
                )}
              >
                {index + 1}
              </span>

              {/* Enable/disable toggle */}
              <motion.button
                onClick={(e) => {
                  e.stopPropagation();
                  handleToggleEffect(effect.id, !effect.enabled);
                }}
                className={cn(
                  'rounded flex-shrink-0',
                  'flex items-center justify-center',
                  'transition-colors duration-150',
                  effect.enabled
                    ? 'bg-green-500 text-slate-900'
                    : 'bg-slate-600/60 text-slate-500',
                  isMobile
                    ? 'w-[18px] h-[18px] text-[0.5rem]'
                    : 'w-5 h-5 text-[0.55rem]'
                )}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                transition={{ type: 'spring', stiffness: 500, damping: 25 }}
                title={effect.enabled ? 'Disable' : 'Enable'}
              >
                {effect.enabled ? '●' : '○'}
              </motion.button>

              {/* Effect name */}
              <span
                className={cn(
                  'flex-1 font-medium',
                  'overflow-hidden text-ellipsis whitespace-nowrap',
                  effect.enabled ? 'text-slate-200' : 'text-slate-500',
                  isMobile ? 'text-[0.7rem]' : 'text-[0.8rem]'
                )}
              >
                {AVAILABLE_EFFECTS.find((e) => e.type === effect.type)?.label || effect.type}
              </span>

              {/* Move buttons */}
              <div
                className="flex flex-col gap-px"
                onClick={(e) => e.stopPropagation()}
              >
                <motion.button
                  onClick={() => handleMoveEffect(effect.id, 'up')}
                  disabled={index === 0}
                  className={cn(
                    'flex items-center justify-center',
                    'rounded-sm bg-slate-600/40',
                    'text-slate-500 transition-colors duration-100',
                    index === 0
                      ? 'opacity-30 cursor-not-allowed'
                      : 'hover:bg-slate-600 hover:text-slate-300',
                    isMobile
                      ? 'w-3.5 h-2.5 text-[0.5rem]'
                      : 'w-4 h-3 text-[0.55rem]'
                  )}
                  whileHover={index !== 0 ? { scale: 1.1 } : undefined}
                  whileTap={index !== 0 ? { scale: 0.9 } : undefined}
                  title="Move up"
                >
                  ▲
                </motion.button>
                <motion.button
                  onClick={() => handleMoveEffect(effect.id, 'down')}
                  disabled={index === effects.length - 1}
                  className={cn(
                    'flex items-center justify-center',
                    'rounded-sm bg-slate-600/40',
                    'text-slate-500 transition-colors duration-100',
                    index === effects.length - 1
                      ? 'opacity-30 cursor-not-allowed'
                      : 'hover:bg-slate-600 hover:text-slate-300',
                    isMobile
                      ? 'w-3.5 h-2.5 text-[0.5rem]'
                      : 'w-4 h-3 text-[0.55rem]'
                  )}
                  whileHover={index !== effects.length - 1 ? { scale: 1.1 } : undefined}
                  whileTap={index !== effects.length - 1 ? { scale: 0.9 } : undefined}
                  title="Move down"
                >
                  ▼
                </motion.button>
              </div>

              {/* Remove button */}
              <motion.button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveEffect(effect.id);
                }}
                className={cn(
                  'flex items-center justify-center flex-shrink-0',
                  'rounded bg-transparent',
                  'text-slate-500 transition-colors duration-150',
                  'hover:bg-red-500/80 hover:text-white',
                  isMobile
                    ? 'w-[18px] h-[18px] text-[0.8rem]'
                    : 'w-5 h-5 text-[0.9rem]'
                )}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                transition={{ type: 'spring', stiffness: 500, damping: 25 }}
                title="Remove effect"
              >
                ×
              </motion.button>
            </motion.div>
          ))
        )}
      </div>

      {/* Effect Editor Panel */}
      <AnimatePresence>
        {selectedEffect && (
          <motion.div
            className={cn(
              'border-t border-slate-700/50',
              isMobile ? 'p-1.5' : 'p-2'
            )}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
          >
            <EffectEditor
              trackId={trackId}
              effectId={selectedEffect.id}
              effectType={selectedEffect.type}
              parameters={selectedEffect.parameters}
              enabled={selectedEffect.enabled}
              onClose={() => setSelectedEffectId(null)}
              onParameterChange={handleParameterChange}
              onEnabledChange={() => syncEffects()}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click outside to close menu */}
      {showAddMenu && (
        <div
          className="fixed inset-0 z-[99]"
          onClick={() => setShowAddMenu(false)}
        />
      )}
    </motion.div>
  );
};

export default EffectRack;
