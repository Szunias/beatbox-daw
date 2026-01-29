/**
 * Note Component
 * Renders a single MIDI note in the piano roll
 */

import React, { useCallback } from 'react';
import { MidiNote } from '../../types/project';

interface NoteProps {
  note: MidiNote;
  x: number;
  y: number;
  width: number;
  height: number;
  isSelected: boolean;
  color?: string;
  onSelect: (noteId: string, additive: boolean) => void;
  onDragStart?: (noteId: string, e: React.MouseEvent) => void;
}

export const Note: React.FC<NoteProps> = ({
  note,
  x,
  y,
  width,
  height,
  isSelected,
  color = 'var(--accent-primary)',
  onSelect,
  onDragStart,
}) => {
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect(note.id, e.shiftKey || e.ctrlKey || e.metaKey);
    },
    [note.id, onSelect]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 0) {
        onDragStart?.(note.id, e);
      }
    },
    [note.id, onDragStart]
  );

  // Calculate velocity-based opacity (higher velocity = more opaque)
  const velocityOpacity = 0.5 + (note.velocity / 127) * 0.5;

  return (
    <div
      className={`piano-roll-note ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width: Math.max(4, width - 1),
        height: Math.max(4, height - 1),
        backgroundColor: color,
        opacity: velocityOpacity,
        borderRadius: 2,
        cursor: 'pointer',
        border: isSelected ? '2px solid white' : '1px solid rgba(0,0,0,0.3)',
        boxShadow: isSelected
          ? '0 0 8px rgba(255,255,255,0.4)'
          : '0 1px 2px rgba(0,0,0,0.2)',
        transition: 'box-shadow 0.1s ease',
        zIndex: isSelected ? 10 : 1,
      }}
      title={`Note: ${note.pitch} | Vel: ${note.velocity} | Dur: ${note.duration}`}
    >
      {/* Resize handle (right edge) */}
      {isSelected && (
        <div
          style={{
            position: 'absolute',
            right: 0,
            top: 0,
            width: 6,
            height: '100%',
            cursor: 'ew-resize',
            backgroundColor: 'rgba(255,255,255,0.3)',
            borderRadius: '0 2px 2px 0',
          }}
        />
      )}
    </div>
  );
};

export default Note;
