/**
 * PianoKeys Component
 * Piano keyboard display on the left side of the piano roll
 */

import React, { useMemo } from 'react';

interface PianoKeysProps {
  topPitch: number;
  bottomPitch: number;
  width?: number;
  noteHeight: number;
  onKeyClick?: (pitch: number) => void;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const isBlackKey = (pitch: number): boolean => {
  const note = pitch % 12;
  return [1, 3, 6, 8, 10].includes(note);
};

const getNoteName = (pitch: number): string => {
  const note = pitch % 12;
  const octave = Math.floor(pitch / 12) - 1;
  return `${NOTE_NAMES[note]}${octave}`;
};

export const PianoKeys: React.FC<PianoKeysProps> = ({
  topPitch,
  bottomPitch,
  width = 60,
  noteHeight,
  onKeyClick,
}) => {
  const keys = useMemo(() => {
    const result: { pitch: number; isBlack: boolean; name: string }[] = [];
    for (let pitch = topPitch; pitch >= bottomPitch; pitch--) {
      result.push({
        pitch,
        isBlack: isBlackKey(pitch),
        name: getNoteName(pitch),
      });
    }
    return result;
  }, [topPitch, bottomPitch]);

  return (
    <div
      className="piano-keys"
      style={{
        width,
        backgroundColor: 'var(--bg-tertiary)',
        borderRight: '1px solid rgba(255,255,255,0.1)',
        overflow: 'hidden',
      }}
    >
      {keys.map((key) => (
        <div
          key={key.pitch}
          onClick={() => onKeyClick?.(key.pitch)}
          style={{
            height: noteHeight,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            paddingRight: 8,
            backgroundColor: key.isBlack ? 'var(--bg-primary)' : 'var(--bg-secondary)',
            borderBottom: '1px solid rgba(255,255,255,0.05)',
            cursor: 'pointer',
            transition: 'background-color 0.1s ease',
            fontSize: '0.65rem',
            color: key.pitch % 12 === 0 ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: key.pitch % 12 === 0 ? 600 : 400,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--accent-secondary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = key.isBlack
              ? 'var(--bg-primary)'
              : 'var(--bg-secondary)';
          }}
        >
          {/* Only show C notes and selected keys */}
          {(key.pitch % 12 === 0 || noteHeight > 12) && (
            <span>{key.name}</span>
          )}
        </div>
      ))}
    </div>
  );
};

export default PianoKeys;
