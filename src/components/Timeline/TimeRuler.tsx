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

  // Drag state for loop region selection and movement
  const [isDragging, setIsDragging] = useState(false);
  const [dragMode, setDragMode] = useState<'create' | 'move' | 'resize-start' | 'resize-end' | null>(null);
  const [dragStartTick, setDragStartTick] = useState<number | null>(null);
  const [dragCurrentTick, setDragCurrentTick] = useState<number | null>(null);
  // For move mode: store the offset from click position to region start
  const [moveOffset, setMoveOffset] = useState<number>(0);
  // Track hover position on loop region (null, 'body', 'left-edge', 'right-edge')
  const [hoverPosition, setHoverPosition] = useState<'body' | 'left-edge' | 'right-edge' | null>(null);
  const rulerRef = useRef<HTMLDivElement>(null);

  // Edge detection zone size in pixels
  const EDGE_ZONE_SIZE = 8;

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

  // Handle hover to show appropriate cursor when over loop region
  const handleHover = useCallback(
    (e: React.MouseEvent) => {
      if (isDragging) return; // Don't change hover state while dragging
      if (!rulerRef.current) return;
      if (!loopRegion.enabled) {
        setHoverPosition(null);
        return;
      }

      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const tick = pixelToTick(x);

      // Calculate pixel positions of loop region edges
      const loopStartX = (loopRegion.startTick - startTick) * pixelsPerTick;
      const loopEndX = (loopRegion.endTick - startTick) * pixelsPerTick;

      // Check if near left edge
      if (Math.abs(x - loopStartX) <= EDGE_ZONE_SIZE) {
        setHoverPosition('left-edge');
      }
      // Check if near right edge
      else if (Math.abs(x - loopEndX) <= EDGE_ZONE_SIZE) {
        setHoverPosition('right-edge');
      }
      // Check if inside loop region body
      else if (tick >= loopRegion.startTick && tick <= loopRegion.endTick) {
        setHoverPosition('body');
      }
      // Outside loop region
      else {
        setHoverPosition(null);
      }
    },
    [isDragging, pixelToTick, loopRegion.enabled, loopRegion.startTick, loopRegion.endTick, startTick, pixelsPerTick, EDGE_ZONE_SIZE]
  );

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoverPosition(null);
    }
  }, [isDragging]);

  // Handle double-click to remove loop region
  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      if (!rulerRef.current) return;
      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const tick = pixelToTick(x);

      // Check if double-click is on the loop region
      if (loopRegion.enabled && tick >= loopRegion.startTick && tick <= loopRegion.endTick) {
        setLoopEnabled(false);
        e.preventDefault();
      }
    },
    [pixelToTick, loopRegion.enabled, loopRegion.startTick, loopRegion.endTick, setLoopEnabled]
  );

  // Handle mouse down - start drag or seek
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!rulerRef.current) return;
      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const tick = pixelToTick(x);
      const snappedTick = snapToBar(tick);

      // Shift+click starts loop region selection (create mode)
      if (e.shiftKey) {
        setIsDragging(true);
        setDragMode('create');
        setDragStartTick(snappedTick);
        setDragCurrentTick(snappedTick);
        e.preventDefault();
      } else if (hoverPosition === 'left-edge') {
        // Click on left edge starts resize-start mode
        setIsDragging(true);
        setDragMode('resize-start');
        setDragStartTick(loopRegion.startTick);
        setDragCurrentTick(loopRegion.startTick);
        e.preventDefault();
      } else if (hoverPosition === 'right-edge') {
        // Click on right edge starts resize-end mode
        setIsDragging(true);
        setDragMode('resize-end');
        setDragStartTick(loopRegion.endTick);
        setDragCurrentTick(loopRegion.endTick);
        e.preventDefault();
      } else if (hoverPosition === 'body') {
        // Click inside loop region body starts move mode
        setIsDragging(true);
        setDragMode('move');
        // Store offset from click position to region start (for smooth dragging)
        setMoveOffset(tick - loopRegion.startTick);
        setDragStartTick(loopRegion.startTick);
        setDragCurrentTick(loopRegion.startTick);
        e.preventDefault();
      } else {
        // Regular click seeks to position
        seekTo(snappedTick);
      }
    },
    [pixelToTick, snapToBar, seekTo, hoverPosition, loopRegion.startTick, loopRegion.endTick]
  );

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !rulerRef.current) return;
      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const tick = pixelToTick(x);

      if (dragMode === 'move') {
        // In move mode, calculate new region start based on offset
        const newStart = snapToBar(Math.max(0, tick - moveOffset));
        setDragCurrentTick(newStart);
      } else if (dragMode === 'resize-start') {
        // In resize-start mode, snap to bar but constrain to valid range
        const snappedTick = snapToBar(Math.max(0, tick));
        // Ensure minimum 1 bar width: new start must be at least 1 bar before current end
        const maxStart = loopRegion.endTick - ticksPerBar;
        setDragCurrentTick(Math.min(snappedTick, maxStart));
      } else if (dragMode === 'resize-end') {
        // In resize-end mode, snap to bar but constrain to valid range
        const snappedTick = snapToBar(Math.max(0, tick));
        // Ensure minimum 1 bar width: new end must be at least 1 bar after current start
        const minEnd = loopRegion.startTick + ticksPerBar;
        setDragCurrentTick(Math.max(snappedTick, minEnd));
      } else {
        // In create mode, snap to bar
        const snappedTick = snapToBar(tick);
        setDragCurrentTick(snappedTick);
      }
    },
    [isDragging, dragMode, pixelToTick, snapToBar, moveOffset, loopRegion.startTick, loopRegion.endTick, ticksPerBar]
  );

  // Handle mouse up - finish drag
  const handleMouseUp = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || dragCurrentTick === null) {
        setIsDragging(false);
        setDragMode(null);
        return;
      }

      if (dragMode === 'move') {
        // Move mode: update region position maintaining width
        const regionWidth = loopRegion.endTick - loopRegion.startTick;
        const newStart = Math.max(0, dragCurrentTick);
        const newEnd = newStart + regionWidth;
        setLoopRegion(newStart, newEnd);
      } else if (dragMode === 'resize-start') {
        // Resize-start mode: update start tick, keep end fixed
        const newStart = Math.max(0, dragCurrentTick);
        // Enforce minimum 1 bar width
        const maxStart = loopRegion.endTick - ticksPerBar;
        const constrainedStart = Math.min(newStart, maxStart);
        setLoopRegion(constrainedStart, loopRegion.endTick);
      } else if (dragMode === 'resize-end') {
        // Resize-end mode: update end tick, keep start fixed
        const newEnd = Math.max(0, dragCurrentTick);
        // Enforce minimum 1 bar width
        const minEnd = loopRegion.startTick + ticksPerBar;
        const constrainedEnd = Math.max(newEnd, minEnd);
        setLoopRegion(loopRegion.startTick, constrainedEnd);
      } else if (dragStartTick !== null) {
        // Create mode: calculate final loop region (ensure start < end)
        const regionStart = Math.min(dragStartTick, dragCurrentTick);
        const regionEnd = Math.max(dragStartTick, dragCurrentTick);

        // Only set loop if region is at least 1 bar
        if (regionEnd - regionStart >= ticksPerBar) {
          setLoopRegion(regionStart, regionEnd);
          setLoopEnabled(true);
        }
      }

      setIsDragging(false);
      setDragMode(null);
      setDragStartTick(null);
      setDragCurrentTick(null);
      setMoveOffset(0);
    },
    [isDragging, dragMode, dragStartTick, dragCurrentTick, ticksPerBar, loopRegion.startTick, loopRegion.endTick, setLoopRegion, setLoopEnabled]
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

  // Get current tick for keyboard shortcuts
  const { currentTick, getInterpolatedTick } = useTransportStore();

  // Keyboard shortcuts for loop control
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle shortcuts when typing in input fields
      const activeElement = document.activeElement;
      if (
        activeElement instanceof HTMLInputElement ||
        activeElement instanceof HTMLTextAreaElement ||
        activeElement instanceof HTMLSelectElement ||
        (activeElement as HTMLElement)?.isContentEditable
      ) {
        return;
      }

      // Get current position (use interpolated tick for smooth position during playback)
      const currentPosition = snapToBar(getInterpolatedTick());

      switch (e.key.toLowerCase()) {
        case 'l':
          // L: Toggle loop enabled
          setLoopEnabled(!loopRegion.enabled);
          e.preventDefault();
          break;
        case '[':
          // [: Set loop start to current position
          // Ensure start is before end (swap if needed or enforce minimum)
          if (currentPosition < loopRegion.endTick - ticksPerBar) {
            setLoopRegion(currentPosition, loopRegion.endTick);
          } else {
            // If current position is at or past end, set start to at least 1 bar before end
            setLoopRegion(loopRegion.endTick - ticksPerBar, loopRegion.endTick);
          }
          // Enable loop if not already enabled
          if (!loopRegion.enabled) {
            setLoopEnabled(true);
          }
          e.preventDefault();
          break;
        case ']':
          // ]: Set loop end to current position
          // Ensure end is after start (enforce minimum 1 bar)
          if (currentPosition > loopRegion.startTick + ticksPerBar) {
            setLoopRegion(loopRegion.startTick, currentPosition);
          } else {
            // If current position is at or before start, set end to at least 1 bar after start
            setLoopRegion(loopRegion.startTick, loopRegion.startTick + ticksPerBar);
          }
          // Enable loop if not already enabled
          if (!loopRegion.enabled) {
            setLoopEnabled(true);
          }
          e.preventDefault();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [loopRegion, setLoopEnabled, setLoopRegion, snapToBar, getInterpolatedTick, ticksPerBar]);

  // Calculate loop region display position
  const loopRegionDisplay = useMemo(() => {
    if (isDragging && dragCurrentTick !== null) {
      if (dragMode === 'move') {
        // Show preview at new position during move
        const regionWidth = loopRegion.endTick - loopRegion.startTick;
        const x = (dragCurrentTick - startTick) * pixelsPerTick;
        const displayWidth = regionWidth * pixelsPerTick;
        return { x, width: displayWidth, active: false };
      } else if (dragMode === 'resize-start') {
        // Show preview with new start position during resize-start
        const x = (dragCurrentTick - startTick) * pixelsPerTick;
        const displayWidth = (loopRegion.endTick - dragCurrentTick) * pixelsPerTick;
        return { x, width: displayWidth, active: false };
      } else if (dragMode === 'resize-end') {
        // Show preview with new end position during resize-end
        const x = (loopRegion.startTick - startTick) * pixelsPerTick;
        const displayWidth = (dragCurrentTick - loopRegion.startTick) * pixelsPerTick;
        return { x, width: displayWidth, active: false };
      } else if (dragStartTick !== null) {
        // Show preview during create drag
        const previewStart = Math.min(dragStartTick, dragCurrentTick);
        const previewEnd = Math.max(dragStartTick, dragCurrentTick);
        const x = (previewStart - startTick) * pixelsPerTick;
        const displayWidth = (previewEnd - previewStart) * pixelsPerTick;
        return { x, width: displayWidth, active: false };
      }
    }
    if (loopRegion.enabled) {
      // Show saved loop region
      const x = (loopRegion.startTick - startTick) * pixelsPerTick;
      const displayWidth = (loopRegion.endTick - loopRegion.startTick) * pixelsPerTick;
      return { x, width: displayWidth, active: true };
    }
    return null;
  }, [isDragging, dragMode, dragStartTick, dragCurrentTick, loopRegion, startTick, pixelsPerTick]);

  // Determine cursor based on state
  const cursor = isDragging
    ? dragMode === 'move'
      ? 'grabbing'
      : dragMode === 'resize-start' || dragMode === 'resize-end'
        ? 'ew-resize'
        : 'col-resize'
    : hoverPosition === 'left-edge' || hoverPosition === 'right-edge'
      ? 'ew-resize'
      : hoverPosition === 'body'
        ? 'grab'
        : 'pointer';

  return (
    <div
      ref={rulerRef}
      className="time-ruler"
      style={{ width, height, position: 'relative', cursor }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleHover}
      onMouseLeave={handleMouseLeave}
      onDoubleClick={handleDoubleClick}
      title="Click to seek • Shift+drag to set loop region • Drag loop to move • Drag edges to resize • Double-click loop to disable"
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
