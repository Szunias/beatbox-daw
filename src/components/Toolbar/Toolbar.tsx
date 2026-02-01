/**
 * Toolbar Component
 * Main DAW toolbar with transport, BPM, time signature, snap settings, and tools
 */

import React, { useCallback } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore, Tool } from '../../stores/uiStore';
import { TransportControls } from './TransportControls';
import { SnapValue } from '../../types/project';

export const Toolbar: React.FC = () => {
  const { project, setBpm, setTimeSignature, setProjectName } = useProjectStore();
  const { snapSettings, setSnapEnabled, setSnapValue, currentTool, setTool } = useUIStore();

  const handleBpmChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseInt(e.target.value, 10);
      if (!isNaN(value) && value >= 20 && value <= 300) {
        setBpm(value);
      }
    },
    [setBpm]
  );

  const handleBpmKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setBpm(project.bpm + 1);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setBpm(project.bpm - 1);
      }
    },
    [project.bpm, setBpm]
  );

  const handleNumeratorChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = parseInt(e.target.value, 10);
      setTimeSignature(value, project.timeSignatureDenominator);
    },
    [project.timeSignatureDenominator, setTimeSignature]
  );

  const handleDenominatorChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = parseInt(e.target.value, 10);
      setTimeSignature(project.timeSignatureNumerator, value);
    },
    [project.timeSignatureNumerator, setTimeSignature]
  );

  const handleSnapValueChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setSnapValue(e.target.value as SnapValue);
    },
    [setSnapValue]
  );

  const tools: { id: Tool; label: string; shortcut: string }[] = [
    { id: 'select', label: 'Select', shortcut: 'V' },
    { id: 'draw', label: 'Draw', shortcut: 'D' },
    { id: 'erase', label: 'Erase', shortcut: 'E' },
    { id: 'slice', label: 'Slice', shortcut: 'S' },
  ];

  return (
    <div
      className="toolbar"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '20px',
        padding: '8px 16px',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        flexWrap: 'wrap',
      }}
    >
      {/* Transport controls */}
      <TransportControls />

      {/* Divider */}
      <div
        style={{
          width: 1,
          height: 30,
          backgroundColor: 'rgba(255,255,255,0.1)',
        }}
      />

      {/* BPM control */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <label
          style={{
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
          }}
        >
          BPM
        </label>
        <input
          type="number"
          min={20}
          max={300}
          value={project.bpm}
          onChange={handleBpmChange}
          onKeyDown={handleBpmKeyDown}
          style={{
            width: 60,
            padding: '4px 8px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            fontSize: '0.9rem',
            fontWeight: 600,
            textAlign: 'center',
          }}
        />
      </div>

      {/* Time signature */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}
      >
        <label
          style={{
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
          }}
        >
          Time
        </label>
        <select
          value={project.timeSignatureNumerator}
          onChange={handleNumeratorChange}
          style={{
            padding: '4px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            fontSize: '0.85rem',
          }}
        >
          {[2, 3, 4, 5, 6, 7, 8].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <span style={{ color: 'var(--text-secondary)' }}>/</span>
        <select
          value={project.timeSignatureDenominator}
          onChange={handleDenominatorChange}
          style={{
            padding: '4px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            fontSize: '0.85rem',
          }}
        >
          {[2, 4, 8, 16].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>

      {/* Divider */}
      <div
        style={{
          width: 1,
          height: 30,
          backgroundColor: 'rgba(255,255,255,0.1)',
        }}
      />

      {/* Snap settings */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <button
          onClick={() => setSnapEnabled(!snapSettings.enabled)}
          title="Toggle Snap"
          style={{
            padding: '4px 8px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: snapSettings.enabled
              ? 'var(--accent-secondary)'
              : 'rgba(255,255,255,0.1)',
            color: snapSettings.enabled ? 'white' : 'var(--text-secondary)',
            cursor: 'pointer',
            fontSize: '0.75rem',
            fontWeight: 600,
          }}
        >
          SNAP
        </button>
        <select
          value={snapSettings.value}
          onChange={handleSnapValueChange}
          disabled={!snapSettings.enabled}
          style={{
            padding: '4px 6px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'var(--bg-primary)',
            color: snapSettings.enabled ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontSize: '0.85rem',
            opacity: snapSettings.enabled ? 1 : 0.5,
          }}
        >
          <option value="none">Off</option>
          <option value="1/1">1 Bar</option>
          <option value="1/2">1/2</option>
          <option value="1/4">1/4</option>
          <option value="1/8">1/8</option>
          <option value="1/16">1/16</option>
          <option value="1/32">1/32</option>
        </select>
      </div>

      {/* Divider */}
      <div
        style={{
          width: 1,
          height: 30,
          backgroundColor: 'rgba(255,255,255,0.1)',
        }}
      />

      {/* Tools */}
      <div
        style={{
          display: 'flex',
          gap: '2px',
        }}
      >
        {tools.map((tool) => (
          <button
            key={tool.id}
            onClick={() => setTool(tool.id)}
            title={`${tool.label} (${tool.shortcut})`}
            style={{
              padding: '6px 10px',
              border: 'none',
              borderRadius: 4,
              backgroundColor:
                currentTool === tool.id ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
              color: currentTool === tool.id ? 'white' : 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 500,
            }}
          >
            {tool.label}
          </button>
        ))}
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Project name */}
      <input
        type="text"
        value={project.name}
        onChange={(e) => setProjectName(e.target.value)}
        style={{
          padding: '4px 12px',
          border: 'none',
          borderRadius: 4,
          backgroundColor: 'var(--bg-primary)',
          color: 'var(--text-primary)',
          fontSize: '0.9rem',
          fontWeight: 500,
          width: 200,
          textAlign: 'right',
        }}
      />
    </div>
  );
};

export default Toolbar;
