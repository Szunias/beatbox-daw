/**
 * TimeRuler Component
 * Displays the time ruler with bar/beat markers at the top of the timeline
 */

import React, { useMemo } from 'react';
import { TICKS_PER_BEAT, ticksToMeasures } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore } from '../../stores/uiStore';

interface TimeRulerProps {
  width: number;
  height?: number;
}

export const TimeRuler: React.FC<TimeRulerProps> = ({ width, height = 30 }) => {
  const { project } = useProjectStore();
  const { timelineViewport } = useUIStore();
  const { bpm, timeSignatureNumerator } = project;
  const { startTick, endTick } = timelineViewport;

  const ticksPerBar = TICKS_PER_BEAT * timeSignatureNumerator;
  const tickRange = endTick - startTick;
  const pixelsPerTick = width / tickRange;

  // Calculate which markers to show based on zoom level
  const markers = useMemo(() => {
    const result: { tick: number; type: 'bar' | 'beat' | 'subdivision'; label?: string }[] = [];

    // Determine granularity based on zoom
    const ticksPerPixel = tickRange / width;
    let step: number;
    let showBeats = false;
    let showSubdivisions = false;

    if (ticksPerPixel < 2) {
      // Very zoomed in: show subdivisions
      step = TICKS_PER_BEAT / 4;
      showBeats = true;
      showSubdivisions = true;
    } else if (ticksPerPixel < 10) {
      // Zoomed in: show beats
      step = TICKS_PER_BEAT;
      showBeats = true;
    } else if (ticksPerPixel < 40) {
      // Medium zoom: show bars
      step = ticksPerBar;
    } else {
      // Zoomed out: show every 4 bars
      step = ticksPerBar * 4;
    }

    // Start from a rounded position
    const start = Math.floor(startTick / step) * step;

    for (let tick = start; tick <= endTick + step; tick += step) {
      if (tick < 0) continue;

      const isBar = tick % ticksPerBar === 0;
      const isBeat = tick % TICKS_PER_BEAT === 0;

      if (isBar) {
        const barNumber = Math.floor(tick / ticksPerBar) + 1;
        result.push({ tick, type: 'bar', label: `${barNumber}` });
      } else if (isBeat && showBeats) {
        const beatInBar = Math.floor((tick % ticksPerBar) / TICKS_PER_BEAT) + 1;
        result.push({ tick, type: 'beat', label: `.${beatInBar}` });
      } else if (showSubdivisions) {
        result.push({ tick, type: 'subdivision' });
      }
    }

    return result;
  }, [startTick, endTick, width, ticksPerBar]);

  return (
    <div className="time-ruler" style={{ width, height, position: 'relative' }}>
      <svg width={width} height={height}>
        {/* Background */}
        <rect x={0} y={0} width={width} height={height} fill="var(--bg-tertiary)" />

        {/* Markers */}
        {markers.map((marker, idx) => {
          const x = (marker.tick - startTick) * pixelsPerTick;
          if (x < -50 || x > width + 50) return null;

          const markerHeight = marker.type === 'bar' ? 15 : marker.type === 'beat' ? 10 : 5;
          const opacity = marker.type === 'bar' ? 1 : marker.type === 'beat' ? 0.7 : 0.4;

          return (
            <g key={`${marker.tick}-${idx}`}>
              <line
                x1={x}
                y1={height}
                x2={x}
                y2={height - markerHeight}
                stroke="var(--text-secondary)"
                strokeWidth={marker.type === 'bar' ? 1 : 0.5}
                opacity={opacity}
              />
              {marker.label && (
                <text
                  x={x + 4}
                  y={12}
                  fill="var(--text-secondary)"
                  fontSize={marker.type === 'bar' ? 11 : 9}
                  fontWeight={marker.type === 'bar' ? 600 : 400}
                >
                  {marker.label}
                </text>
              )}
            </g>
          );
        })}

        {/* Bottom border */}
        <line
          x1={0}
          y1={height - 0.5}
          x2={width}
          y2={height - 0.5}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={1}
        />
      </svg>
    </div>
  );
};

export default TimeRuler;
