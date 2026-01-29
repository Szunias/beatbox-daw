/**
 * PianoRoll Component
 * Main piano roll editor for MIDI clip editing
 */

import React, { useCallback, useRef, useState, useEffect } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore } from '../../stores/uiStore';
import { PianoKeys } from './PianoKeys';
import { NoteGrid } from './NoteGrid';
import { MidiClip, MidiNote, createMidiNote, TICKS_PER_BEAT, getSnapTicks } from '../../types/project';

const PIANO_KEYS_WIDTH = 60;
const DEFAULT_NOTE_HEIGHT = 16;

interface PianoRollProps {
  height?: number;
}

export const PianoRoll: React.FC<PianoRollProps> = ({ height = 300 }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(600);

  const { project, getClip, addNote, removeNote, updateNote } = useProjectStore();
  const {
    pianoRollTrackId,
    pianoRollClipId,
    pianoRollViewport,
    selectedNoteIds,
    selectNotes,
    addToNoteSelection,
    clearNoteSelection,
    snapSettings,
    closePianoRoll,
    setPianoRollViewport,
    scrollPianoRoll,
    scrollPianoRollVertical,
    zoomPianoRoll,
  } = useUIStore();

  // Get current clip
  const clip = pianoRollTrackId && pianoRollClipId
    ? getClip(pianoRollTrackId, pianoRollClipId)
    : null;

  const midiClip = clip && (clip.type === 'midi' || clip.type === 'beatbox')
    ? (clip as MidiClip)
    : null;

  // Update container width on resize
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth - PIANO_KEYS_WIDTH);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Update viewport when clip changes
  useEffect(() => {
    if (midiClip) {
      setPianoRollViewport({
        startTick: midiClip.startTick,
        endTick: midiClip.startTick + midiClip.duration,
      });
    }
  }, [midiClip?.id]);

  const { startTick, endTick, topPitch, bottomPitch } = pianoRollViewport;
  const pitchRange = topPitch - bottomPitch + 1;
  const gridHeight = pitchRange * DEFAULT_NOTE_HEIGHT;

  // Handle wheel for scroll/zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();

      if (e.ctrlKey || e.metaKey) {
        // Zoom
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        zoomPianoRoll(zoomFactor);
      } else if (e.shiftKey) {
        // Horizontal scroll
        const tickRange = endTick - startTick;
        const scrollAmount = (e.deltaY / containerWidth) * tickRange * 0.5;
        scrollPianoRoll(scrollAmount);
      } else {
        // Vertical scroll (pitch)
        const pitchDelta = e.deltaY > 0 ? -2 : 2;
        scrollPianoRollVertical(pitchDelta);
      }
    },
    [endTick, startTick, containerWidth, zoomPianoRoll, scrollPianoRoll, scrollPianoRollVertical]
  );

  // Handle note selection
  const handleSelectNote = useCallback(
    (noteId: string, additive: boolean) => {
      if (additive) {
        addToNoteSelection([noteId]);
      } else {
        selectNotes([noteId]);
      }
    },
    [selectNotes, addToNoteSelection]
  );

  // Handle adding new note
  const handleAddNote = useCallback(
    (pitch: number, tick: number) => {
      if (!pianoRollTrackId || !pianoRollClipId) return;

      const snapTicks = getSnapTicks(snapSettings.value);
      const duration = snapTicks; // Default note length = snap value

      const newNote = createMidiNote(pitch, 100, tick, duration);
      addNote(pianoRollTrackId, pianoRollClipId, newNote);
      selectNotes([newNote.id]);
    },
    [pianoRollTrackId, pianoRollClipId, snapSettings.value, addNote, selectNotes]
  );

  // Handle deselect
  const handleDeselectAll = useCallback(() => {
    clearNoteSelection();
  }, [clearNoteSelection]);

  // Handle key preview (play note when clicking on piano keys)
  const handleKeyClick = useCallback((pitch: number) => {
    // TODO: Send MIDI note preview to engine
    console.log('Preview note:', pitch);
  }, []);

  // Handle delete key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedNoteIds.length > 0 && pianoRollTrackId && pianoRollClipId) {
          selectedNoteIds.forEach((noteId) => {
            removeNote(pianoRollTrackId, pianoRollClipId, noteId);
          });
          clearNoteSelection();
        }
      } else if (e.key === 'Escape') {
        closePianoRoll();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNoteIds, pianoRollTrackId, pianoRollClipId, removeNote, clearNoteSelection, closePianoRoll]);

  if (!midiClip) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: 8,
          color: 'var(--text-secondary)',
        }}
      >
        Double-click a MIDI clip to edit it
      </div>
    );
  }

  return (
    <div
      className="piano-roll-container"
      ref={containerRef}
      style={{
        height,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          backgroundColor: 'var(--bg-tertiary)',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: midiClip.color,
            }}
          />
          <span style={{ fontWeight: 500, fontSize: '0.9rem' }}>
            {midiClip.name}
          </span>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
            ({midiClip.data.notes.length} notes)
          </span>
        </div>
        <button
          onClick={closePianoRoll}
          style={{
            padding: '4px 8px',
            border: 'none',
            borderRadius: 4,
            backgroundColor: 'rgba(255,255,255,0.1)',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            fontSize: '0.8rem',
          }}
        >
          Close
        </button>
      </div>

      {/* Main content */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          overflow: 'hidden',
        }}
        onWheel={handleWheel}
      >
        {/* Piano keys */}
        <PianoKeys
          topPitch={topPitch}
          bottomPitch={bottomPitch}
          width={PIANO_KEYS_WIDTH}
          noteHeight={DEFAULT_NOTE_HEIGHT}
          onKeyClick={handleKeyClick}
        />

        {/* Note grid */}
        <div
          style={{
            flex: 1,
            overflow: 'auto',
          }}
        >
          <NoteGrid
            notes={midiClip.data.notes}
            startTick={startTick}
            endTick={endTick}
            topPitch={topPitch}
            bottomPitch={bottomPitch}
            width={containerWidth}
            height={gridHeight}
            noteHeight={DEFAULT_NOTE_HEIGHT}
            snapValue={snapSettings.value}
            selectedNoteIds={selectedNoteIds}
            color={midiClip.color}
            onSelectNote={handleSelectNote}
            onAddNote={handleAddNote}
            onDeselectAll={handleDeselectAll}
          />
        </div>
      </div>

      {/* Footer with info */}
      <div
        style={{
          padding: '4px 12px',
          backgroundColor: 'var(--bg-tertiary)',
          borderTop: '1px solid rgba(255,255,255,0.1)',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)',
          display: 'flex',
          gap: '16px',
        }}
      >
        <span>Snap: {snapSettings.value}</span>
        <span>Selected: {selectedNoteIds.length}</span>
        <span>Click to add note | Del to delete | Esc to close</span>
      </div>
    </div>
  );
};

export default PianoRoll;
