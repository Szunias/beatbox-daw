/**
 * TimeRuler Component
 * Displays the time ruler with bar/beat markers at the top of the timeline
 * Supports click-to-seek and drag-to-select loop region
 */

import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { TICKS_PER_BEAT } from '../../types/project';
import { useProjectStore } from '../../stores/projectStore';
import { useUIStore } from '../../stores/uiStore';
import { useTransportStore } from '../../stores/transportStore';

interface TimeRulerProps {
  width: number;
  height?: number;
}

export const TimeRuler: React.FC<TimeRulerProps> = ({ width, height = 30 }) => {
  const { project } = useProjectStore();
  const { timelineViewport } = useUIStore();
  const { loopRegion, setLoopRegion, setLoopEnabled, seekTo } = useTransportStore();
  const { bpm, timeSignatureNumerator } = project;
  const { startTick, endTick } = timelineViewport;

  // Drag state for loop region selection
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartTick, setDragStartTick] = useState<number | null>(null);
  const [dragCurrentTick, setDragCurrentTick] = useState<number | null>(null);
  const rulerRef = useRef<HTMLDivElement>(null);

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

  // Convert pixel position to tick
  const pixelToTick = useCallback(
    (pixelX: number): number => {
      return Math.max(0, startTick + pixelX / pixelsPerTick);
    },
    [startTick, pixelsPerTick]
  );

  // Snap tick to bar boundary for cleaner loop regions
  const snapToBar = useCallback(
    (tick: number): number => {
      return Math.round(tick / ticksPerBar) * ticksPerBar;
    },
    [ticksPerBar]
  );

  // Handle mouse down - start drag or seek
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!rulerRef.current) return;
      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const tick = pixelToTick(x);
      const snappedTick = snapToBar(tick);

      // Shift+click starts loop region selection
      if (e.shiftKey) {
        setIsDragging(true);
        setDragStartTick(snappedTick);
        setDragCurrentTick(snappedTick);
        e.preventDefault();
      } else {
        // Regular click seeks to position
        seekTo(snappedTick);
      }
    },
    [pixelToTick, snapToBar, seekTo]
  );

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !rulerRef.current) return;
      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const tick = pixelToTick(x);
      const snappedTick = snapToBar(tick);
      setDragCurrentTick(snappedTick);
    },
    [isDragging, pixelToTick, snapToBar]
  );

  // Handle mouse up - finish drag
  const handleMouseUp = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || dragStartTick === null || dragCurrentTick === null) {
        setIsDragging(false);
        return;
      }

      // Calculate final loop region (ensure start < end)
      const regionStart = Math.min(dragStartTick, dragCurrentTick);
      const regionEnd = Math.max(dragStartTick, dragCurrentTick);

      // Only set loop if region is at least 1 bar
      if (regionEnd - regionStart >= ticksPerBar) {
        setLoopRegion(regionStart, regionEnd);
        setLoopEnabled(true);
      }

      setIsDragging(false);
      setDragStartTick(null);
      setDragCurrentTick(null);
    },
    [isDragging, dragStartTick, dragCurrentTick, ticksPerBar, setLoopRegion, setLoopEnabled]
  );

  // Attach global mouse events for drag tracking
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Calculate loop region display position
  const loopRegionDisplay = useMemo(() => {
    if (isDragging && dragStartTick !== null && dragCurrentTick !== null) {
      // Show preview during drag
      const previewStart = Math.min(dragStartTick, dragCurrentTick);
      const previewEnd = Math.max(dragStartTick, dragCurrentTick);
      const x = (previewStart - startTick) * pixelsPerTick;
      const displayWidth = (previewEnd - previewStart) * pixelsPerTick;
      return { x, width: displayWidth, active: false };
    } else if (loopRegion.enabled) {
      // Show saved loop region
      const x = (loopRegion.startTick - startTick) * pixelsPerTick;
      const displayWidth = (loopRegion.endTick - loopRegion.startTick) * pixelsPerTick;
      return { x, width: displayWidth, active: true };
    }
    return null;
  }, [isDragging, dragStartTick, dragCurrentTick, loopRegion, startTick, pixelsPerTick]);

  return (
    <div
      ref={rulerRef}
      className="time-ruler"
      style={{ width, height, position: 'relative', cursor: 'pointer' }}
      onMouseDown={handleMouseDown}
      title="Click to seek • Shift+drag to set loop region"
    >
      <svg width={width} height={height}>
        {/* Background */}
        <rect x={0} y={0} width={width} height={height} fill="var(--bg-tertiary)" />

        {/* Loop Region Overlay */}
        {loopRegionDisplay && loopRegionDisplay.width > 0 && (
          <rect
            x={loopRegionDisplay.x}
            y={0}
            width={loopRegionDisplay.width}
            height={height}
            fill={loopRegionDisplay.active ? 'rgba(74, 222, 128, 0.3)' : 'rgba(74, 222, 128, 0.2)'}
            stroke={loopRegionDisplay.active ? 'var(--accent-primary)' : 'rgba(74, 222, 128, 0.5)'}
            strokeWidth={loopRegionDisplay.active ? 2 : 1}
          />
        )}

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

        {/* Visual hint for loop selection when no loop is set */}
        {!loopRegion.enabled && !isDragging && (
          <text
            x={width - 8}
            y={height / 2 + 4}
            fill="var(--text-secondary)"
            fontSize={9}
            fontStyle="italic"
            opacity={0.5}
            textAnchor="end"
          >
            Shift+drag for loop
          </text>
        )}
      </svg>
    </div>
  );
};

export default TimeRuler;
