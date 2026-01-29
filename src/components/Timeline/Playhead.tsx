/**
 * Playhead Component
 * Displays the playback position indicator on the timeline
 */

import React from 'react';
import { useTransportStore } from '../../stores/transportStore';
import { useUIStore } from '../../stores/uiStore';

interface PlayheadProps {
  height: number;
  containerWidth: number;
}

export const Playhead: React.FC<PlayheadProps> = ({ height, containerWidth }) => {
  const { currentTick, state } = useTransportStore();
  const { timelineViewport } = useUIStore();
  const { startTick, endTick } = timelineViewport;

  const tickRange = endTick - startTick;
  const pixelsPerTick = containerWidth / tickRange;
  const x = (currentTick - startTick) * pixelsPerTick;

  // Don't render if outside visible area
  if (x < -10 || x > containerWidth + 10) {
    return null;
  }

  const isPlaying = state === 'playing' || state === 'recording';
  const isRecording = state === 'recording';

  return (
    <div
      className="playhead"
      style={{
        position: 'absolute',
        left: x,
        top: 0,
        height,
        width: 2,
        backgroundColor: isRecording ? 'var(--error)' : 'var(--accent-primary)',
        pointerEvents: 'none',
        zIndex: 100,
        transition: isPlaying ? 'none' : 'left 0.05s ease-out',
      }}
    >
      {/* Playhead handle */}
      <div
        style={{
          position: 'absolute',
          top: -8,
          left: -6,
          width: 0,
          height: 0,
          borderLeft: '7px solid transparent',
          borderRight: '7px solid transparent',
          borderTop: `10px solid ${isRecording ? 'var(--error)' : 'var(--accent-primary)'}`,
        }}
      />
    </div>
  );
};

export default Playhead;
