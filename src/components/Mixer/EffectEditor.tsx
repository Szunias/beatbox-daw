/**
 * EffectEditor Component
 * UI for editing individual effect parameters
 * Wires parameter changes to AudioEngine for real-time audio processing
 */

import React, { useCallback, useState, useEffect } from 'react';
import { getAudioEngine } from '../../audio';
import type {
  EffectType,
  EffectParameters,
  EQ3BandParameters,
  CompressorParameters,
  DelayParameters,
  ReverbParameters,
} from '../../audio';

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
    <div
      className="effect-editor"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        padding: isMobile ? '8px' : '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: isMobile ? '8px' : '12px',
        minWidth: isMobile ? 200 : 260,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          paddingBottom: isMobile ? '6px' : '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={handleEnabledToggle}
            style={{
              width: isMobile ? 24 : 28,
              height: isMobile ? 24 : 28,
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              backgroundColor: enabled ? 'var(--success)' : 'var(--bg-tertiary)',
              color: enabled ? 'var(--bg-primary)' : 'var(--text-secondary)',
              fontSize: isMobile ? '0.65rem' : '0.75rem',
              fontWeight: 700,
              transition: 'background-color 0.1s ease',
            }}
            title={enabled ? 'Disable effect' : 'Enable effect'}
          >
            {enabled ? 'ON' : 'OFF'}
          </button>
          <span
            style={{
              fontSize: isMobile ? '0.85rem' : '0.95rem',
              fontWeight: 600,
              color: enabled ? 'var(--text-primary)' : 'var(--text-secondary)',
            }}
          >
            {effectName}
          </span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            style={{
              width: isMobile ? 22 : 26,
              height: isMobile ? 22 : 26,
              border: 'none',
              borderRadius: '50%',
              cursor: 'pointer',
              backgroundColor: 'var(--bg-tertiary)',
              color: 'var(--text-secondary)',
              fontSize: isMobile ? '0.8rem' : '0.9rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title="Close editor"
          >
            ×
          </button>
        )}
      </div>

      {/* Parameters */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: isMobile ? '6px' : '10px',
          opacity: enabled ? 1 : 0.5,
        }}
      >
        {Object.entries(paramConfigs).map(([paramName, config]) => {
          const value = getParamValue(paramName);

          return (
            <div
              key={paramName}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span
                  style={{
                    fontSize: isMobile ? '0.7rem' : '0.75rem',
                    color: 'var(--text-secondary)',
                    fontWeight: 500,
                  }}
                >
                  {config.label}
                </span>
                <span
                  style={{
                    fontSize: isMobile ? '0.65rem' : '0.7rem',
                    color: 'var(--text-primary)',
                    fontFamily: 'monospace',
                    backgroundColor: 'var(--bg-tertiary)',
                    padding: '2px 6px',
                    borderRadius: 3,
                  }}
                >
                  {formatValue(paramName, value)}
                </span>
              </div>
              <input
                type="range"
                min={config.min}
                max={config.max}
                step={config.step}
                value={value}
                onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
                disabled={!enabled}
                style={{
                  width: '100%',
                  height: isMobile ? 24 : 8,
                  cursor: enabled ? 'pointer' : 'not-allowed',
                  accentColor: 'var(--accent-primary)',
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default EffectEditor;
