/**
 * Track Component
 * Renders a single track lane in the timeline with its clips
 */

import React, { useMemo, useCallback } from 'react';
import { Track as TrackType, TICKS_PER_BEAT, createMidiClip } from '../../types/project';
import { useUIStore } from '../../stores/uiStore';
import { useProjectStore } from '../../stores/projectStore';
import { useTransportStore } from '../../stores/transportStore';
import { Clip } from './Clip';

interface TrackProps {
  track: TrackType;
  height: number;
  width: number;
}

export const Track: React.FC<TrackProps> = ({ track, height, width }) => {
  const { timelineViewport, openPianoRoll, snapSettings } = useUIStore();
  const { project, addClip } = useProjectStore();
  const { isRecording } = useTransportStore();
  const { startTick, endTick } = timelineViewport;

  const tickRange = endTick - startTick;
  const pixelsPerTick = width / tickRange;

  // Handle double-click to create new clip and open piano roll
  const handleDoubleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    // Only for MIDI/drum tracks
    if (track.type !== 'midi' && track.type !== 'drum') return;

    // Calculate tick position from click
    const rect = e.currentTarget.getBoundingClientRect();
    const relativeX = e.clientX - rect.left;
    let clickTick = startTick + (relativeX / width) * tickRange;

    // Snap to grid
    if (snapSettings.enabled) {
      const snapTicks = snapSettings.value === '1/1' ? TICKS_PER_BEAT * 4 :
                        snapSettings.value === '1/2' ? TICKS_PER_BEAT * 2 :
                        snapSettings.value === '1/4' ? TICKS_PER_BEAT :
                        snapSettings.value === '1/8' ? TICKS_PER_BEAT / 2 :
                        snapSettings.value === '1/16' ? TICKS_PER_BEAT / 4 :
                        snapSettings.value === '1/32' ? TICKS_PER_BEAT / 8 : TICKS_PER_BEAT;
      clickTick = Math.round(clickTick / snapTicks) * snapTicks;
    }

    // Create new clip (1 bar duration)
    const clipDuration = TICKS_PER_BEAT * 4;
    const newClip = createMidiClip(
      `New Clip`,
      Math.max(0, clickTick),
      clipDuration,
      [],
      track.color
    );

    addClip(track.id, newClip);
    openPianoRoll(track.id, newClip.id);
  }, [track.id, track.type, track.color, startTick, tickRange, width, snapSettings, addClip, openPianoRoll]);

  // Background grid
  const gridLines = useMemo(() => {
    const lines: { x: number; type: 'bar' | 'beat' }[] = [];
    const ticksPerBar = TICKS_PER_BEAT * project.timeSignatureNumerator;
    const ticksPerPixel = tickRange / width;

    // Determine grid granularity based on zoom
    let step: number;
    if (ticksPerPixel < 10) {
      step = TICKS_PER_BEAT; // Show beats
    } else {
      step = ticksPerBar; // Show bars only
    }

    const start = Math.floor(startTick / step) * step;

    for (let tick = start; tick <= endTick + step; tick += step) {
      if (tick < 0) continue;
      const x = (tick - startTick) * pixelsPerTick;
      if (x < -1 || x > width + 1) continue;

      const isBar = tick % ticksPerBar === 0;
      lines.push({ x, type: isBar ? 'bar' : 'beat' });
    }

    return lines;
  }, [startTick, endTick, width, project.timeSignatureNumerator]);

  // Filter clips that are visible in viewport
  const visibleClips = useMemo(() => {
    return track.clips.filter((clip) => {
      const clipEnd = clip.startTick + clip.duration;
      return clipEnd >= startTick && clip.startTick <= endTick;
    });
  }, [track.clips, startTick, endTick]);

  return (
    <div
      className="track"
      onDoubleClick={handleDoubleClick}
      style={{
        height,
        width,
        position: 'relative',
        backgroundColor: 'var(--bg-primary)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        borderLeft: track.armed ? '3px solid var(--error)' : 'none',
        boxSizing: 'border-box',
      }}
    >
      {/* Grid background */}
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
        {gridLines.map((line, idx) => (
          <line
            key={idx}
            x1={line.x}
            y1={0}
            x2={line.x}
            y2={height}
            stroke={
              line.type === 'bar'
                ? 'rgba(255,255,255,0.1)'
                : 'rgba(255,255,255,0.04)'
            }
            strokeWidth={line.type === 'bar' ? 1 : 0.5}
          />
        ))}
      </svg>

      {/* Clips */}
      {visibleClips.map((clip) => (
        <Clip
          key={clip.id}
          clip={clip}
          trackId={track.id}
          trackHeight={height}
          pixelsPerTick={pixelsPerTick}
          startTickOffset={startTick}
        />
      ))}

      {/* Track mute overlay */}
      {track.muted && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.4)',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Armed track recording indicator */}
      {track.armed && isRecording && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(229, 62, 62, 0.1)',
            pointerEvents: 'none',
            animation: 'armed-track-pulse 1.5s ease-in-out infinite',
          }}
        >
          <style>
            {`
              @keyframes armed-track-pulse {
                0%, 100% { background-color: rgba(229, 62, 62, 0.1); }
                50% { background-color: rgba(229, 62, 62, 0.2); }
              }
            `}
          </style>
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: 8,
              transform: 'translateY(-50%)',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              padding: '2px 6px',
              backgroundColor: 'var(--error)',
              borderRadius: 3,
              fontSize: '0.65rem',
              fontWeight: 600,
              color: 'white',
            }}
          >
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                backgroundColor: 'white',
                animation: 'recording-blink 0.8s ease-in-out infinite',
              }}
            />
            <style>
              {`
                @keyframes recording-blink {
                  0%, 100% { opacity: 1; }
                  50% { opacity: 0.3; }
                }
              `}
            </style>
            REC
          </div>
        </div>
      )}

      {/* Armed track indicator (not recording) */}
      {track.armed && !isRecording && (
        <div
          style={{
            position: 'absolute',
            top: 4,
            left: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: '2px 6px',
            backgroundColor: 'rgba(229, 62, 62, 0.8)',
            borderRadius: 3,
            fontSize: '0.65rem',
            fontWeight: 600,
            color: 'white',
            pointerEvents: 'none',
          }}
        >
          ARMED
        </div>
      )}
    </div>
  );
};

export default Track;
