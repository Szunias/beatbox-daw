/**
 * EffectRack Component
 * Manages effect chain for a track with add/remove/reorder functionality
 * Wires to AudioEngine's EffectsProcessor for real-time audio processing
 */

import React, { useState, useCallback, useEffect } from 'react';
import { getAudioEngine } from '../../audio';
import type { EffectType, EffectParameters } from '../../audio';
import { EffectEditor } from './EffectEditor';

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
    <div
      className="effect-rack"
      style={{
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-primary)',
        borderRadius: 8,
        overflow: 'hidden',
        minWidth: isMobile ? 200 : 280,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: isMobile ? '6px 8px' : '8px 12px',
          backgroundColor: 'var(--bg-tertiary)',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              width: 4,
              height: isMobile ? 14 : 16,
              backgroundColor: trackColor,
              borderRadius: 2,
            }}
          />
          <span
            style={{
              fontSize: isMobile ? '0.8rem' : '0.9rem',
              fontWeight: 600,
              color: 'var(--text-primary)',
            }}
          >
            Effects
          </span>
          <span
            style={{
              fontSize: isMobile ? '0.65rem' : '0.75rem',
              color: 'var(--text-secondary)',
              backgroundColor: 'var(--bg-primary)',
              padding: '2px 6px',
              borderRadius: 8,
            }}
          >
            {effects.length}/{maxEffects}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          {/* Bypass button */}
          <button
            onClick={handleToggleBypass}
            style={{
              padding: isMobile ? '3px 6px' : '4px 8px',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              fontSize: isMobile ? '0.6rem' : '0.7rem',
              fontWeight: 600,
              backgroundColor: bypassed ? 'var(--warning)' : 'var(--bg-secondary)',
              color: bypassed ? 'var(--bg-primary)' : 'var(--text-secondary)',
              transition: 'background-color 0.1s ease',
            }}
            title={bypassed ? 'Enable effects' : 'Bypass all effects'}
          >
            BYP
          </button>
          {/* Add button */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowAddMenu(!showAddMenu)}
              disabled={effects.length >= maxEffects}
              style={{
                padding: isMobile ? '3px 6px' : '4px 8px',
                border: 'none',
                borderRadius: 4,
                cursor: effects.length >= maxEffects ? 'not-allowed' : 'pointer',
                fontSize: isMobile ? '0.7rem' : '0.8rem',
                fontWeight: 600,
                backgroundColor: 'var(--accent-primary)',
                color: 'var(--text-primary)',
                opacity: effects.length >= maxEffects ? 0.5 : 1,
              }}
              title="Add effect"
            >
              +
            </button>
            {/* Add effect menu */}
            {showAddMenu && (
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  right: 0,
                  marginTop: 4,
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: 4,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                  zIndex: 100,
                  overflow: 'hidden',
                  minWidth: 100,
                }}
              >
                {AVAILABLE_EFFECTS.map((effect) => (
                  <button
                    key={effect.type}
                    onClick={() => handleAddEffect(effect.type)}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: isMobile ? '6px 10px' : '8px 12px',
                      border: 'none',
                      backgroundColor: 'transparent',
                      color: 'var(--text-primary)',
                      fontSize: isMobile ? '0.75rem' : '0.85rem',
                      textAlign: 'left',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    {effect.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Effect slots */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
          padding: isMobile ? '4px' : '6px',
          minHeight: 60,
          opacity: bypassed ? 0.5 : 1,
        }}
      >
        {effects.length === 0 ? (
          <div
            style={{
              padding: isMobile ? '12px' : '16px',
              textAlign: 'center',
              color: 'var(--text-secondary)',
              fontSize: isMobile ? '0.75rem' : '0.85rem',
            }}
          >
            No effects - click + to add
          </div>
        ) : (
          effects.map((effect, index) => (
            <div
              key={effect.id}
              onClick={() => setSelectedEffectId(effect.id === selectedEffectId ? null : effect.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: isMobile ? '4px 6px' : '6px 8px',
                backgroundColor:
                  effect.id === selectedEffectId ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
                borderRadius: 4,
                cursor: 'pointer',
                border:
                  effect.id === selectedEffectId
                    ? '1px solid var(--accent-primary)'
                    : '1px solid transparent',
                transition: 'background-color 0.1s ease, border-color 0.1s ease',
              }}
            >
              {/* Effect number */}
              <span
                style={{
                  fontSize: isMobile ? '0.6rem' : '0.65rem',
                  color: 'var(--text-secondary)',
                  width: 14,
                  textAlign: 'center',
                }}
              >
                {index + 1}
              </span>

              {/* Enable/disable toggle */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleToggleEffect(effect.id, !effect.enabled);
                }}
                style={{
                  width: isMobile ? 18 : 20,
                  height: isMobile ? 18 : 20,
                  border: 'none',
                  borderRadius: 3,
                  cursor: 'pointer',
                  backgroundColor: effect.enabled ? 'var(--success)' : 'var(--bg-tertiary)',
                  color: effect.enabled ? 'var(--bg-primary)' : 'var(--text-secondary)',
                  fontSize: isMobile ? '0.5rem' : '0.55rem',
                  fontWeight: 700,
                  flexShrink: 0,
                }}
                title={effect.enabled ? 'Disable' : 'Enable'}
              >
                {effect.enabled ? '●' : '○'}
              </button>

              {/* Effect name */}
              <span
                style={{
                  flex: 1,
                  fontSize: isMobile ? '0.7rem' : '0.8rem',
                  fontWeight: 500,
                  color: effect.enabled ? 'var(--text-primary)' : 'var(--text-secondary)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {AVAILABLE_EFFECTS.find((e) => e.type === effect.type)?.label || effect.type}
              </span>

              {/* Move buttons */}
              <div
                style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => handleMoveEffect(effect.id, 'up')}
                  disabled={index === 0}
                  style={{
                    width: isMobile ? 14 : 16,
                    height: isMobile ? 10 : 12,
                    border: 'none',
                    borderRadius: 2,
                    cursor: index === 0 ? 'not-allowed' : 'pointer',
                    backgroundColor: 'var(--bg-tertiary)',
                    color: 'var(--text-secondary)',
                    fontSize: isMobile ? '0.5rem' : '0.55rem',
                    opacity: index === 0 ? 0.3 : 0.7,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  title="Move up"
                >
                  ▲
                </button>
                <button
                  onClick={() => handleMoveEffect(effect.id, 'down')}
                  disabled={index === effects.length - 1}
                  style={{
                    width: isMobile ? 14 : 16,
                    height: isMobile ? 10 : 12,
                    border: 'none',
                    borderRadius: 2,
                    cursor: index === effects.length - 1 ? 'not-allowed' : 'pointer',
                    backgroundColor: 'var(--bg-tertiary)',
                    color: 'var(--text-secondary)',
                    fontSize: isMobile ? '0.5rem' : '0.55rem',
                    opacity: index === effects.length - 1 ? 0.3 : 0.7,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  title="Move down"
                >
                  ▼
                </button>
              </div>

              {/* Remove button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveEffect(effect.id);
                }}
                style={{
                  width: isMobile ? 18 : 20,
                  height: isMobile ? 18 : 20,
                  border: 'none',
                  borderRadius: 3,
                  cursor: 'pointer',
                  backgroundColor: 'transparent',
                  color: 'var(--text-secondary)',
                  fontSize: isMobile ? '0.8rem' : '0.9rem',
                  flexShrink: 0,
                }}
                title="Remove effect"
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--error)';
                  e.currentTarget.style.color = 'var(--text-primary)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                }}
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>

      {/* Effect Editor Panel */}
      {selectedEffect && (
        <div
          style={{
            borderTop: '1px solid rgba(255,255,255,0.1)',
            padding: isMobile ? '6px' : '8px',
          }}
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
        </div>
      )}

      {/* Click outside to close menu */}
      {showAddMenu && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 99,
          }}
          onClick={() => setShowAddMenu(false)}
        />
      )}
    </div>
  );
};

export default EffectRack;
