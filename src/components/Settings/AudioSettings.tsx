/**
 * Audio Settings Component
 * Allows selection of audio input/output devices
 */

import React, { useEffect, useState } from 'react';
import { useWebSocket, AudioDevice } from '../../hooks/useWebSocket';

interface AudioSettingsProps {
  onClose?: () => void;
}

export const AudioSettings: React.FC<AudioSettingsProps> = ({ onClose }) => {
  const {
    isConnected,
    devices,
    listDevices,
    setAudioDevice,
    sendMessage,
  } = useWebSocket();

  const [inputDevices, setInputDevices] = useState<AudioDevice[]>([]);
  const [outputDevices, setOutputDevices] = useState<AudioDevice[]>([]);
  const [selectedInputId, setSelectedInputId] = useState<number | null>(null);
  const [selectedOutputId, setSelectedOutputId] = useState<number | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Request device list on mount and when connected
  useEffect(() => {
    if (isConnected) {
      listDevices();
    }
  }, [isConnected, listDevices]);

  // Parse devices into input/output
  useEffect(() => {
    const inputs = devices.filter(d => d.type === 'input' || d.type === 'both');
    const outputs = devices.filter(d => d.type === 'output' || d.type === 'both');
    setInputDevices(inputs);
    setOutputDevices(outputs);
  }, [devices]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    listDevices();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const deviceId = parseInt(e.target.value, 10);
    setSelectedInputId(deviceId);
    setAudioDevice(deviceId, 'input');
  };

  const handleOutputChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const deviceId = parseInt(e.target.value, 10);
    setSelectedOutputId(deviceId);
    setAudioDevice(deviceId, 'output');
  };

  return (
    <div
      style={{
        padding: '16px',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        minWidth: 300,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
        }}
      >
        <h3
          style={{
            margin: 0,
            fontSize: '1rem',
            color: 'var(--text-primary)',
          }}
        >
          Audio Settings
        </h3>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={handleRefresh}
            disabled={!isConnected || isRefreshing}
            style={{
              padding: '4px 10px',
              border: 'none',
              borderRadius: 4,
              backgroundColor: 'rgba(255,255,255,0.1)',
              color: 'var(--text-secondary)',
              cursor: isConnected ? 'pointer' : 'not-allowed',
              fontSize: '0.75rem',
              opacity: isRefreshing ? 0.5 : 1,
            }}
          >
            {isRefreshing ? 'Refreshing...' : 'Refresh'}
          </button>
          {onClose && (
            <button
              onClick={onClose}
              style={{
                padding: '4px 10px',
                border: 'none',
                borderRadius: 4,
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '0.75rem',
              }}
            >
              Close
            </button>
          )}
        </div>
      </div>

      {/* Connection status */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '16px',
          padding: '8px',
          backgroundColor: isConnected ? 'rgba(74, 222, 128, 0.1)' : 'rgba(248, 113, 113, 0.1)',
          borderRadius: 4,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: isConnected ? '#4ade80' : '#f87171',
          }}
        />
        <span
          style={{
            fontSize: '0.8rem',
            color: isConnected ? '#4ade80' : '#f87171',
          }}
        >
          {isConnected ? 'Engine Connected' : 'Engine Disconnected'}
        </span>
      </div>

      {/* Input Device Selection */}
      <div style={{ marginBottom: '16px' }}>
        <label
          style={{
            display: 'block',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
            marginBottom: '6px',
          }}
        >
          Input Device (Microphone)
        </label>
        <select
          value={selectedInputId ?? ''}
          onChange={handleInputChange}
          disabled={!isConnected || inputDevices.length === 0}
          style={{
            width: '100%',
            padding: '8px 12px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            fontSize: '0.9rem',
            cursor: isConnected ? 'pointer' : 'not-allowed',
          }}
        >
          <option value="">
            {inputDevices.length === 0 ? 'No devices found' : 'Select input device...'}
          </option>
          {inputDevices.map((device) => (
            <option key={device.id} value={device.id}>
              {device.name} ({device.channels}ch, {device.sample_rate}Hz)
            </option>
          ))}
        </select>
      </div>

      {/* Output Device Selection */}
      <div style={{ marginBottom: '16px' }}>
        <label
          style={{
            display: 'block',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
            marginBottom: '6px',
          }}
        >
          Output Device (Speakers)
        </label>
        <select
          value={selectedOutputId ?? ''}
          onChange={handleOutputChange}
          disabled={!isConnected || outputDevices.length === 0}
          style={{
            width: '100%',
            padding: '8px 12px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            fontSize: '0.9rem',
            cursor: isConnected ? 'pointer' : 'not-allowed',
          }}
        >
          <option value="">
            {outputDevices.length === 0 ? 'No devices found' : 'Select output device...'}
          </option>
          {outputDevices.map((device) => (
            <option key={device.id} value={device.id}>
              {device.name} ({device.channels}ch, {device.sample_rate}Hz)
            </option>
          ))}
        </select>
      </div>

      {/* Info text */}
      <p
        style={{
          fontSize: '0.75rem',
          color: 'var(--text-secondary)',
          margin: 0,
          lineHeight: 1.4,
        }}
      >
        Select your microphone for beatbox detection and speakers for audio playback.
        Changes take effect immediately.
      </p>
    </div>
  );
};

export default AudioSettings;
