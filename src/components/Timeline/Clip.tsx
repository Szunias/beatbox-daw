/**
 * Clip Component
 * Renders a MIDI or audio clip on the timeline
 */

import React, { useCallback, useMemo } from 'react';
import { Clip as ClipType, MidiClip, TICKS_PER_BEAT } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore } from '../../stores/uiStore';

interface ClipProps {
  clip: ClipType;
  trackId: string;
  trackHeight: number;
  pixelsPerTick: number;
  startTickOffset: number;
}

export const Clip: React.FC<ClipProps> = ({
  clip,
  trackId,
  trackHeight,
  pixelsPerTick,
  startTickOffset,
}) => {
  const { selectClip, selectedClipId } = useProjectStore();
  const { openPianoRoll } = useUIStore();

  const isSelected = selectedClipId === clip.id;

  const x = (clip.startTick - startTickOffset) * pixelsPerTick;
  const width = Math.max(10, clip.duration * pixelsPerTick);

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    selectClip(clip.id);
  }, [clip.id, selectClip]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (clip.type === 'midi' || clip.type === 'beatbox') {
      openPianoRoll(trackId, clip.id);
    }
  }, [clip.type, trackId, clip.id, openPianoRoll]);

  // Render mini notes preview for MIDI clips
  const notesPreview = useMemo(() => {
    if (clip.type !== 'midi' && clip.type !== 'beatbox') return null;

    const midiClip = clip as MidiClip;
    const notes = midiClip.data.notes;
    if (notes.length === 0) return null;

    // Find pitch range for scaling
    const pitches = notes.map((n) => n.pitch);
    const minPitch = Math.min(...pitches);
    const maxPitch = Math.max(...pitches);
    const pitchRange = Math.max(1, maxPitch - minPitch);

    const clipInnerHeight = trackHeight - 20; // Account for header/padding

    return notes.map((note) => {
      const noteX = (note.startTick / clip.duration) * width;
      const noteWidth = Math.max(2, (note.duration / clip.duration) * width);
      const noteY = clipInnerHeight - ((note.pitch - minPitch) / pitchRange) * clipInnerHeight;
      const noteHeight = Math.max(2, clipInnerHeight / Math.max(12, pitchRange));

      return (
        <rect
          key={note.id}
          x={noteX}
          y={noteY + 14} // Offset for clip header
          width={noteWidth - 1}
          height={noteHeight}
          fill="rgba(255,255,255,0.6)"
          rx={1}
        />
      );
    });
  }, [clip, width, trackHeight]);

  return (
    <div
      className={`clip ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      style={{
        position: 'absolute',
        left: x,
        top: 2,
        width,
        height: trackHeight - 4,
        backgroundColor: clip.muted ? 'rgba(100,100,100,0.5)' : clip.color,
        borderRadius: 4,
        overflow: 'hidden',
        cursor: 'pointer',
        opacity: clip.muted ? 0.5 : 1,
        border: isSelected ? '2px solid white' : '1px solid rgba(0,0,0,0.3)',
        boxShadow: isSelected
          ? '0 0 10px rgba(255,255,255,0.3)'
          : '0 1px 3px rgba(0,0,0,0.3)',
        transition: 'box-shadow 0.15s ease, border 0.15s ease',
      }}
    >
      {/* Clip header */}
      <div
        style={{
          height: 14,
          backgroundColor: 'rgba(0,0,0,0.2)',
          padding: '0 4px',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}
      >
        <span
          style={{
            fontSize: '0.65rem',
            fontWeight: 500,
            color: 'rgba(255,255,255,0.9)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {clip.name}
        </span>
        {clip.muted && (
          <span
            style={{
              fontSize: '0.6rem',
              color: 'rgba(255,255,255,0.6)',
            }}
          >
            (M)
          </span>
        )}
      </div>

      {/* Notes preview */}
      {(clip.type === 'midi' || clip.type === 'beatbox') && (
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
          {notesPreview}
        </svg>
      )}

      {/* Resize handles (visual only for now) */}
      {isSelected && (
        <>
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              width: 4,
              height: '100%',
              cursor: 'ew-resize',
              backgroundColor: 'rgba(255,255,255,0.3)',
            }}
          />
          <div
            style={{
              position: 'absolute',
              right: 0,
              top: 0,
              width: 4,
              height: '100%',
              cursor: 'ew-resize',
              backgroundColor: 'rgba(255,255,255,0.3)',
            }}
          />
        </>
      )}
    </div>
  );
};

export default Clip;
