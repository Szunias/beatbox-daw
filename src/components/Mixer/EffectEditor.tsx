/**
 * EffectEditor Component
 * UI for editing individual effect parameters
 * Wires parameter changes to AudioEngine for real-time audio processing
 *
 * Features modern glassmorphism design with Motion animations
 */

import React, { useCallback, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { getAudioEngine } from '../../audio';
import type {
  EffectType,
  EffectParameters,
  EQ3BandParameters,
  CompressorParameters,
  DelayParameters,
  ReverbParameters,
} from '../../audio';
import { cn } from '../../lib/utils';

interface EffectEditorProps {
  trackId: string;
  effectId: string;
  effectType: EffectType;
  parameters: EffectParameters;
  enabled: boolean;
  onClose?: () => void;
  onParameterChange?: (paramName: string, value: number) => void;
  onEnabledChange?: (enabled: boolean) => void;
}

// Effect display names
const EFFECT_NAMES: Record<EffectType, string> = {
  eq3band: '3-Band EQ',
  compressor: 'Compressor',
  delay: 'Delay',
  reverb: 'Reverb',
};

// Effect icons (Unicode symbols)
const EFFECT_ICONS: Record<EffectType, string> = {
  eq3band: '◐',
  compressor: '◉',
  delay: '◎',
  reverb: '◈',
};

// Parameter configurations
interface ParamConfig {
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  format?: (value: number) => string;
}

const EQ_PARAMS: Record<string, ParamConfig> = {
  lowFreq: { label: 'Low Freq', min: 20, max: 500, step: 1, unit: 'Hz' },
  highFreq: { label: 'High Freq', min: 1000, max: 16000, step: 100, unit: 'Hz' },
  lowGain: { label: 'Low Gain', min: -12, max: 12, step: 0.1, unit: 'dB' },
  midGain: { label: 'Mid Gain', min: -12, max: 12, step: 0.1, unit: 'dB' },
  highGain: { label: 'High Gain', min: -12, max: 12, step: 0.1, unit: 'dB' },
  mix: { label: 'Mix', min: 0, max: 1, step: 0.01, format: (v) => `${Math.round(v * 100)}%` },
};

const COMPRESSOR_PARAMS: Record<string, ParamConfig> = {
  threshold: { label: 'Threshold', min: -60, max: 0, step: 1, unit: 'dB' },
  ratio: { label: 'Ratio', min: 1, max: 20, step: 0.1, format: (v) => `${v.toFixed(1)}:1` },
  attack: { label: 'Attack', min: 0.001, max: 1, step: 0.001, unit: 's' },
  release: { label: 'Release', min: 0.01, max: 1, step: 0.01, unit: 's' },
  knee: { label: 'Knee', min: 0, max: 40, step: 1, unit: 'dB' },
  mix: { label: 'Mix', min: 0, max: 1, step: 0.01, format: (v) => `${Math.round(v * 100)}%` },
};

const DELAY_PARAMS: Record<string, ParamConfig> = {
  delayTime: { label: 'Time', min: 0.001, max: 2, step: 0.001, unit: 's' },
  feedback: { label: 'Feedback', min: 0, max: 0.95, step: 0.01, format: (v) => `${Math.round(v * 100)}%` },
  mix: { label: 'Mix', min: 0, max: 1, step: 0.01, format: (v) => `${Math.round(v * 100)}%` },
};

const REVERB_PARAMS: Record<string, ParamConfig> = {
  decay: { label: 'Decay', min: 0.1, max: 10, step: 0.1, unit: 's' },
  preDelay: { label: 'Pre-Delay', min: 0, max: 0.1, step: 0.001, unit: 's' },
  mix: { label: 'Mix', min: 0, max: 1, step: 0.01, format: (v) => `${Math.round(v * 100)}%` },
};

const EFFECT_PARAM_CONFIGS: Record<EffectType, Record<string, ParamConfig>> = {
  eq3band: EQ_PARAMS,
  compressor: COMPRESSOR_PARAMS,
  delay: DELAY_PARAMS,
  reverb: REVERB_PARAMS,
};

export const EffectEditor: React.FC<EffectEditorProps> = ({
  trackId,
  effectId,
  effectType,
  parameters,
  enabled,
  onClose,
  onParameterChange,
  onEnabledChange,
}) => {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleParameterChange = useCallback(
    (paramName: string, value: number) => {
      const audioEngine = getAudioEngine();
      audioEngine.setTrackEffectParameter(trackId, effectId, paramName, value);
      onParameterChange?.(paramName, value);
    },
    [trackId, effectId, onParameterChange]
  );

  const handleEnabledToggle = useCallback(() => {
    const audioEngine = getAudioEngine();
    const newEnabled = !enabled;
    audioEngine.setTrackEffectEnabled(trackId, effectId, newEnabled);
    onEnabledChange?.(newEnabled);
  }, [trackId, effectId, enabled, onEnabledChange]);

  const paramConfigs = EFFECT_PARAM_CONFIGS[effectType];
  const effectName = EFFECT_NAMES[effectType];
  const effectIcon = EFFECT_ICONS[effectType];

  const formatValue = (paramName: string, value: number): string => {
    const config = paramConfigs[paramName];
    if (!config) return String(value);
    if (config.format) return config.format(value);
    if (config.unit) return `${value.toFixed(paramName === 'ratio' ? 1 : config.step < 1 ? 2 : 0)}${config.unit}`;
    return String(value);
  };

  // Helper to get parameter value with proper typing
  const getParamValue = (paramName: string): number => {
    return (parameters as unknown as Record<string, number>)[paramName] ?? 0;
  };

  return (
    <motion.div
      className={cn(
        'effect-editor',
        // Glassmorphism effect
        'bg-slate-800/70 backdrop-blur-lg',
        'border border-slate-700/50',
        'rounded-xl shadow-xl shadow-black/30',
        // Layout
        'flex flex-col',
        isMobile ? 'min-w-[200px] gap-2 p-2' : 'min-w-[260px] gap-3 p-3'
      )}
      initial={{ opacity: 0, scale: 0.95, y: -5 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: -5 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between',
          'border-b border-slate-700/50',
          isMobile ? 'pb-2' : 'pb-2.5'
        )}
      >
        <div className="flex items-center gap-2">
          {/* Enable/Disable toggle */}
          <motion.button
            onClick={handleEnabledToggle}
            className={cn(
              'font-bold rounded',
              'border border-transparent',
              'flex items-center justify-center',
              'transition-colors duration-150',
              enabled
                ? 'bg-green-500 text-slate-900 border-green-400 shadow-md shadow-green-500/30'
                : 'bg-slate-700/60 text-slate-500 hover:bg-slate-700 hover:text-slate-400',
              isMobile
                ? 'w-[26px] h-[26px] text-[0.65rem]'
                : 'w-[30px] h-[30px] text-[0.75rem]'
            )}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 500, damping: 25 }}
            title={enabled ? 'Disable effect' : 'Enable effect'}
          >
            {enabled ? 'ON' : 'OFF'}
          </motion.button>

          {/* Effect icon and name */}
          <div className="flex items-center gap-1.5">
            <span
              className={cn(
                'transition-colors duration-200',
                enabled ? 'text-rose-400' : 'text-slate-600',
                isMobile ? 'text-base' : 'text-lg'
              )}
            >
              {effectIcon}
            </span>
            <span
              className={cn(
                'font-semibold transition-colors duration-200',
                enabled ? 'text-slate-100' : 'text-slate-500',
                isMobile ? 'text-[0.85rem]' : 'text-[0.95rem]'
              )}
            >
              {effectName}
            </span>
          </div>
        </div>

        {/* Close button */}
        {onClose && (
          <motion.button
            onClick={onClose}
            className={cn(
              'flex items-center justify-center',
              'rounded-full',
              'bg-slate-700/60 text-slate-400',
              'transition-colors duration-150',
              'hover:bg-slate-600 hover:text-slate-200',
              isMobile
                ? 'w-[22px] h-[22px] text-[0.85rem]'
                : 'w-[26px] h-[26px] text-[0.95rem]'
            )}
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            transition={{ type: 'spring', stiffness: 400, damping: 20 }}
            title="Close editor"
          >
            ×
          </motion.button>
        )}
      </div>

      {/* Parameters */}
      <motion.div
        className={cn(
          'flex flex-col',
          'transition-opacity duration-200',
          enabled ? 'opacity-100' : 'opacity-40',
          isMobile ? 'gap-2' : 'gap-3'
        )}
        animate={{ opacity: enabled ? 1 : 0.4 }}
        transition={{ duration: 0.2 }}
      >
        {Object.entries(paramConfigs).map(([paramName, config], index) => {
          const value = getParamValue(paramName);

          return (
            <motion.div
              key={paramName}
              className="flex flex-col gap-1"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.03, duration: 0.15 }}
            >
              {/* Parameter label and value */}
              <div className="flex justify-between items-center">
                <span
                  className={cn(
                    'font-medium text-slate-400',
                    isMobile ? 'text-[0.7rem]' : 'text-[0.75rem]'
                  )}
                >
                  {config.label}
                </span>
                <span
                  className={cn(
                    'font-mono',
                    'bg-slate-900/60 backdrop-blur-sm',
                    'border border-slate-700/50',
                    'rounded px-1.5 py-0.5',
                    'text-slate-200',
                    isMobile ? 'text-[0.6rem]' : 'text-[0.7rem]'
                  )}
                >
                  {formatValue(paramName, value)}
                </span>
              </div>

              {/* Slider */}
              <input
                type="range"
                min={config.min}
                max={config.max}
                step={config.step}
                value={value}
                onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
                disabled={!enabled}
                className={cn(
                  'w-full cursor-pointer',
                  'appearance-none bg-transparent',
                  // Track styling
                  '[&::-webkit-slider-runnable-track]:h-1.5',
                  '[&::-webkit-slider-runnable-track]:rounded-full',
                  '[&::-webkit-slider-runnable-track]:bg-gradient-to-r',
                  '[&::-webkit-slider-runnable-track]:from-slate-700',
                  '[&::-webkit-slider-runnable-track]:to-slate-600',
                  // Thumb styling
                  '[&::-webkit-slider-thumb]:appearance-none',
                  '[&::-webkit-slider-thumb]:w-3.5',
                  '[&::-webkit-slider-thumb]:h-3.5',
                  '[&::-webkit-slider-thumb]:rounded-full',
                  '[&::-webkit-slider-thumb]:bg-gradient-to-b',
                  '[&::-webkit-slider-thumb]:from-rose-400',
                  '[&::-webkit-slider-thumb]:to-rose-500',
                  '[&::-webkit-slider-thumb]:shadow-md',
                  '[&::-webkit-slider-thumb]:shadow-rose-500/30',
                  '[&::-webkit-slider-thumb]:mt-[-4px]',
                  '[&::-webkit-slider-thumb]:transition-all',
                  '[&::-webkit-slider-thumb]:duration-150',
                  '[&::-webkit-slider-thumb]:hover:scale-125',
                  '[&::-webkit-slider-thumb]:hover:from-rose-300',
                  '[&::-webkit-slider-thumb]:hover:to-rose-400',
                  '[&::-webkit-slider-thumb]:active:from-rose-500',
                  '[&::-webkit-slider-thumb]:active:to-rose-600',
                  // Disabled state
                  !enabled && 'cursor-not-allowed opacity-50',
                  !enabled && '[&::-webkit-slider-thumb]:from-slate-500',
                  !enabled && '[&::-webkit-slider-thumb]:to-slate-600',
                  !enabled && '[&::-webkit-slider-thumb]:shadow-none',
                  // Height
                  isMobile ? 'h-6' : 'h-5'
                )}
              />
            </motion.div>
          );
        })}
      </motion.div>

      {/* Footer hint */}
      <div
        className={cn(
          'text-center text-slate-600',
          'border-t border-slate-700/30 pt-2',
          isMobile ? 'text-[0.6rem]' : 'text-[0.65rem]'
        )}
      >
        {enabled ? 'Drag sliders to adjust' : 'Enable effect to adjust parameters'}
      </div>
    </motion.div>
  );
};

export default EffectEditor;
