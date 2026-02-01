/**
 * Timeline Component
 * Main timeline/arrangement view containing all tracks and time ruler
 */

import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useTransportStore } from '../../stores/transportStore';
import { useUIStore } from '../../stores/uiStore';
import { TimeRuler } from './TimeRuler';
import { Track } from './Track';
import { TrackHeader } from './TrackHeader';
import { Playhead } from './Playhead';
import { TICKS_PER_BEAT } from '../../types/project';
import { ContextMenu, MenuItem } from '../ContextMenu';
import { useContextMenu } from '../../hooks/useContextMenu';

const trackHeaderWidth_DESKTOP = 180;
const trackHeaderWidth_MOBILE = 70;
const TRACK_HEIGHT = 70;
const TIME_RULER_HEIGHT = 30;

interface TimelineProps {
  height?: number;
}

export const Timeline: React.FC<TimelineProps> = ({ height = 400 }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  const trackHeaderWidth = isMobile ? trackHeaderWidth_MOBILE : trackHeaderWidth_DESKTOP;

  const { project, addTrack, removeTrack, selectedTrackId, selectTrack } = useProjectStore();
  const { seekTo, currentTick, stop, isRecording } = useTransportStore();
  const { timelineViewport, scrollTimeline, zoomTimeline, setTimelineViewport } = useUIStore();

  // Context menu
  const { menuState, closeMenu, handlers: contextMenuHandlers } = useContextMenu();

  // Build context menu items
  const contextMenuItems: MenuItem[] = useMemo(() => {
    const items: MenuItem[] = [
      {
        label: 'Add Drum Track',
        onClick: () => addTrack('drum'),
      },
      {
        label: 'Add MIDI Track',
        onClick: () => addTrack('midi'),
      },
      {
        label: 'Add Audio Track',
        onClick: () => addTrack('audio'),
      },
    ];

    // Add track-specific options if a track is selected
    if (selectedTrackId) {
      items.push({ label: '', onClick: () => {}, divider: true });
      items.push({
        label: 'Delete Track',
        onClick: () => {
          removeTrack(selectedTrackId);
        },
      });
    }

    items.push({ label: '', onClick: () => {}, divider: true });
    items.push({
      label: 'Stop Playback',
      onClick: () => stop(),
    });

    return items;
  }, [addTrack, removeTrack, selectedTrackId, stop]);

  // Update container width and mobile state on resize
  useEffect(() => {
    const updateDimensions = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      const headerWidth = mobile ? trackHeaderWidth_MOBILE : trackHeaderWidth_DESKTOP;
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth - headerWidth);
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Handle scroll and zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();

      if (e.ctrlKey || e.metaKey) {
        // Zoom
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const rect = containerRef.current?.getBoundingClientRect();
        if (rect) {
          const relativeX = e.clientX - rect.left - trackHeaderWidth;
          const tickRange = timelineViewport.endTick - timelineViewport.startTick;
          const centerTick =
            timelineViewport.startTick + (relativeX / containerWidth) * tickRange;
          zoomTimeline(zoomFactor, centerTick);
        }
      } else if (e.shiftKey) {
        // Horizontal scroll
        const tickRange = timelineViewport.endTick - timelineViewport.startTick;
        const scrollAmount = (e.deltaY / containerWidth) * tickRange * 0.5;
        scrollTimeline(scrollAmount);
      } else {
        // Default: horizontal scroll
        const tickRange = timelineViewport.endTick - timelineViewport.startTick;
        const scrollAmount = (e.deltaY / containerWidth) * tickRange * 0.3;
        scrollTimeline(scrollAmount);
      }
    },
    [timelineViewport, containerWidth, zoomTimeline, scrollTimeline, trackHeaderWidth]
  );

  // Handle click on timeline to seek
  const handleTimelineClick = useCallback(
    (e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;

      const relativeX = e.clientX - rect.left - trackHeaderWidth;
      if (relativeX < 0) return;

      const tickRange = timelineViewport.endTick - timelineViewport.startTick;
      const clickedTick = timelineViewport.startTick + (relativeX / containerWidth) * tickRange;
      seekTo(Math.max(0, Math.round(clickedTick)));
    },
    [timelineViewport, containerWidth, seekTo, trackHeaderWidth]
  );

  // Handle add track button
  const handleAddTrack = useCallback(
    (type: 'midi' | 'audio' | 'drum') => {
      addTrack(type);
    },
    [addTrack]
  );

  const tracksAreaHeight = height - TIME_RULER_HEIGHT;
  const totalTracksHeight = project.tracks.length * TRACK_HEIGHT;

  return (
    <div
      className="timeline-container"
      ref={containerRef}
      style={{
        display: 'flex',
        flexDirection: 'column',
        height,
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: 8,
        overflow: 'hidden',
      }}
      {...contextMenuHandlers}
    >
      {/* Top row: empty corner + time ruler */}
      <div style={{ display: 'flex', height: TIME_RULER_HEIGHT }}>
        {/* Empty corner */}
        <div
          style={{
            width: trackHeaderWidth,
            backgroundColor: 'var(--bg-tertiary)',
            borderBottom: '1px solid rgba(255,255,255,0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
          }}
        >
          <button
            onClick={() => handleAddTrack('drum')}
            title="Add Drum Track"
            style={{
              padding: '2px 6px',
              fontSize: '0.7rem',
              border: 'none',
              borderRadius: 3,
              backgroundColor: 'var(--accent-primary)',
              color: 'white',
              cursor: 'pointer',
            }}
          >
            +Drum
          </button>
          <button
            onClick={() => handleAddTrack('midi')}
            title="Add MIDI Track"
            style={{
              padding: '2px 6px',
              fontSize: '0.7rem',
              border: 'none',
              borderRadius: 3,
              backgroundColor: 'var(--success)',
              color: 'var(--bg-primary)',
              cursor: 'pointer',
            }}
          >
            +MIDI
          </button>
          <button
            onClick={() => handleAddTrack('audio')}
            title="Add Audio Track"
            style={{
              padding: '2px 6px',
              fontSize: '0.7rem',
              border: 'none',
              borderRadius: 3,
              backgroundColor: 'var(--warning)',
              color: 'var(--bg-primary)',
              cursor: 'pointer',
            }}
          >
            +Audio
          </button>
        </div>

        {/* Time ruler */}
        <TimeRuler width={containerWidth} height={TIME_RULER_HEIGHT} />
      </div>

      {/* Main area: track headers + tracks */}
      <div
        style={{
          display: 'flex',
          flex: 1,
          overflow: 'hidden',
        }}
      >
        {/* Track headers */}
        <div
          style={{
            width: trackHeaderWidth,
            overflowY: 'auto',
            overflowX: 'hidden',
            backgroundColor: 'var(--bg-secondary)',
          }}
        >
          {project.tracks.map((track) => (
            <TrackHeader
              key={track.id}
              track={track}
              isSelected={track.id === selectedTrackId}
              height={TRACK_HEIGHT}
            />
          ))}

          {/* Empty state */}
          {project.tracks.length === 0 && (
            <div
              style={{
                padding: 20,
                textAlign: 'center',
                color: 'var(--text-secondary)',
                fontSize: '0.85rem',
              }}
            >
              No tracks yet.
              <br />
              Click +Drum, +MIDI, or +Audio to add a track.
            </div>
          )}
        </div>

        {/* Tracks area */}
        <div
          style={{
            flex: 1,
            position: 'relative',
            overflow: 'hidden',
          }}
          onWheel={handleWheel}
          onClick={handleTimelineClick}
        >
          {/* Tracks */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              overflowY: 'auto',
              height: tracksAreaHeight,
            }}
          >
            {project.tracks.map((track) => (
              <Track
                key={track.id}
                track={track}
                height={TRACK_HEIGHT}
                width={containerWidth}
              />
            ))}

            {/* Empty track area background */}
            {project.tracks.length === 0 && (
              <div
                style={{
                  height: tracksAreaHeight,
                  backgroundColor: 'var(--bg-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-secondary)',
                  fontSize: '0.9rem',
                }}
              >
                Record some beatbox to create your first clip!
              </div>
            )}
          </div>

          {/* Playhead */}
          <Playhead height={tracksAreaHeight} containerWidth={containerWidth} />

          {/* Recording overlay */}
          {isRecording && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                border: '2px solid var(--error)',
                borderRadius: 4,
                pointerEvents: 'none',
                animation: 'recording-pulse 1s ease-in-out infinite',
              }}
            >
              <style>
                {`
                  @keyframes recording-pulse {
                    0%, 100% { box-shadow: inset 0 0 20px rgba(229, 62, 62, 0.2); }
                    50% { box-shadow: inset 0 0 30px rgba(229, 62, 62, 0.4); }
                  }
                `}
              </style>
              <div
                style={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '4px 8px',
                  backgroundColor: 'var(--error)',
                  borderRadius: 4,
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: 'white',
                }}
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: 'white',
                    animation: 'recording-dot-blink 1s ease-in-out infinite',
                  }}
                />
                <style>
                  {`
                    @keyframes recording-dot-blink {
                      0%, 100% { opacity: 1; }
                      50% { opacity: 0.4; }
                    }
                  `}
                </style>
                REC
              </div>
            </div>
          )}

          {/* Loop region overlay */}
          {/* TODO: Add loop region visualization */}
        </div>
      </div>

      {/* Context Menu */}
      {menuState.isOpen && (
        <ContextMenu
          x={menuState.x}
          y={menuState.y}
          items={contextMenuItems}
          onClose={closeMenu}
        />
      )}
    </div>
  );
};

export default Timeline;
