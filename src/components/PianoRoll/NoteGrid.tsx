/**
 * NoteGrid Component
 * Grid background and note container for the piano roll
 */

import React, { useMemo, useCallback } from 'react';
import { MidiNote, TICKS_PER_BEAT, getSnapTicks, SnapValue } from '../../types/project';
import { Note } from './Note';

interface NoteGridProps {
  notes: MidiNote[];
  startTick: number;
  endTick: number;
  topPitch: number;
  bottomPitch: number;
  width: number;
  height: number;
  noteHeight: number;
  snapValue: SnapValue;
  selectedNoteIds: string[];
  color?: string;
  onSelectNote: (noteId: string, additive: boolean) => void;
  onAddNote?: (pitch: number, startTick: number) => void;
  onDeselectAll?: () => void;
}

const isBlackKey = (pitch: number): boolean => {
  const note = pitch % 12;
  return [1, 3, 6, 8, 10].includes(note);
};

export const NoteGrid: React.FC<NoteGridProps> = ({
  notes,
  startTick,
  endTick,
  topPitch,
  bottomPitch,
  width,
  height,
  noteHeight,
  snapValue,
  selectedNoteIds,
  color = 'var(--accent-primary)',
  onSelectNote,
  onAddNote,
  onDeselectAll,
}) => {
  const tickRange = endTick - startTick;
  const pixelsPerTick = width / tickRange;
  const pitchRange = topPitch - bottomPitch + 1;

  // Calculate grid lines
  const gridLines = useMemo(() => {
    const lines: { x: number; isBeat: boolean }[] = [];
    const snapTicks = getSnapTicks(snapValue);
    const step = Math.max(snapTicks, TICKS_PER_BEAT / 4); // At least 16th notes

    const start = Math.floor(startTick / step) * step;
    for (let tick = start; tick <= endTick + step; tick += step) {
      const x = (tick - startTick) * pixelsPerTick;
      if (x < 0 || x > width) continue;

      const isBeat = tick % TICKS_PER_BEAT === 0;
      lines.push({ x, isBeat });
    }

    return lines;
  }, [startTick, endTick, width, snapValue]);

  // Calculate horizontal lines (per pitch)
  const horizontalLines = useMemo(() => {
    const lines: { y: number; isBlack: boolean; isC: boolean }[] = [];
    for (let pitch = topPitch; pitch >= bottomPitch; pitch--) {
      const y = (topPitch - pitch) * noteHeight;
      lines.push({
        y,
        isBlack: isBlackKey(pitch),
        isC: pitch % 12 === 0,
      });
    }
    return lines;
  }, [topPitch, bottomPitch, noteHeight]);

  // Filter visible notes
  const visibleNotes = useMemo(() => {
    return notes.filter((note) => {
      const noteEnd = note.startTick + note.duration;
      return (
        noteEnd >= startTick &&
        note.startTick <= endTick &&
        note.pitch >= bottomPitch &&
        note.pitch <= topPitch
      );
    });
  }, [notes, startTick, endTick, topPitch, bottomPitch]);

  // Handle click to add note with improved grid snapping
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target !== e.currentTarget) return;

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Calculate pitch from click position
      // Clicking anywhere within a pitch row assigns that pitch
      const rawPitch = topPitch - Math.floor(y / noteHeight);
      // Clamp pitch to valid range for edge case clicks
      const pitch = Math.max(bottomPitch, Math.min(topPitch, rawPitch));

      // Calculate tick from click position
      let tick = startTick + (x / width) * tickRange;

      // Snap to nearest grid intersection
      // The rounding creates a tolerance zone - clicks within half a snap
      // interval of a grid line will snap to that line, creating a forgiving
      // hit area that doesn't require exact clicks on the grid
      const snapTicks = getSnapTicks(snapValue);
      tick = Math.round(tick / snapTicks) * snapTicks;

      // Clamp tick to valid visible range for edge case clicks
      tick = Math.max(startTick, Math.min(endTick, tick));

      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        // Modifier keys: just deselect
        onDeselectAll?.();
      } else if (onAddNote && pitch >= bottomPitch && pitch <= topPitch) {
        // Add note at snapped grid position
        onAddNote(pitch, tick);
      } else {
        onDeselectAll?.();
      }
    },
    [topPitch, bottomPitch, noteHeight, startTick, endTick, tickRange, width, snapValue, onAddNote, onDeselectAll]
  );

  return (
    <div
      className="note-grid"
      style={{
        position: 'relative',
        width,
        height,
        backgroundColor: 'var(--bg-primary)',
        overflow: 'hidden',
        cursor: 'crosshair',
      }}
      onClick={handleClick}
    >
      {/* Background grid */}
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      >
        {/* Horizontal lines (pitch rows) */}
        {horizontalLines.map((line, idx) => (
          <g key={`h-${idx}`}>
            {/* Row background for black keys */}
            {line.isBlack && (
              <rect
                x={0}
                y={line.y}
                width={width}
                height={noteHeight}
                fill="rgba(0,0,0,0.15)"
              />
            )}
            {/* Row border */}
            <line
              x1={0}
              y1={line.y + noteHeight}
              x2={width}
              y2={line.y + noteHeight}
              stroke={line.isC ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.05)'}
              strokeWidth={line.isC ? 1 : 0.5}
            />
          </g>
        ))}

        {/* Vertical lines (time grid) */}
        {gridLines.map((line, idx) => (
          <line
            key={`v-${idx}`}
            x1={line.x}
            y1={0}
            x2={line.x}
            y2={height}
            stroke={line.isBeat ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.05)'}
            strokeWidth={line.isBeat ? 1 : 0.5}
          />
        ))}
      </svg>

      {/* Notes */}
      {visibleNotes.map((note) => {
        const x = (note.startTick - startTick) * pixelsPerTick;
        const y = (topPitch - note.pitch) * noteHeight;
        const noteWidth = note.duration * pixelsPerTick;

        return (
          <Note
            key={note.id}
            note={note}
            x={x}
            y={y}
            width={noteWidth}
            height={noteHeight}
            isSelected={selectedNoteIds.includes(note.id)}
            color={color}
            onSelect={onSelectNote}
          />
        );
      })}
    </div>
  );
};

export default NoteGrid;
