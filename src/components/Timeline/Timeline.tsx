/**
 * Timeline Component
 * Main timeline/arrangement view containing all tracks and time ruler
 */

import React, { useRef, useCallback, useEffect, useState } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useTransportStore } from '../../stores/transportStore';
import { useUIStore } from '../../stores/uiStore';
import { TimeRuler } from './TimeRuler';
import { Track } from './Track';
import { TrackHeader } from './TrackHeader';
import { Playhead } from './Playhead';
import { TICKS_PER_BEAT } from '../../types/project';

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

  const { project, addTrack, selectedTrackId } = useProjectStore();
  const { seekTo, currentTick } = useTransportStore();
  const { timelineViewport, scrollTimeline, zoomTimeline, setTimelineViewport } = useUIStore();

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
              Click +Drum or +MIDI to add a track.
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

          {/* Loop region overlay */}
          {/* TODO: Add loop region visualization */}
        </div>
      </div>
    </div>
  );
};

export default Timeline;
