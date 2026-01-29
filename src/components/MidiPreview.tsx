import React, { useMemo } from 'react';
import { DrumEvent } from '../hooks/useWebSocket';

interface MidiPreviewProps {
  events: DrumEvent[];
  windowDuration?: number; // seconds to show
}

const DRUM_ROWS = ['kick', 'snare', 'hihat', 'clap', 'tom'];

export const MidiPreview: React.FC<MidiPreviewProps> = ({
  events,
  windowDuration = 8,
}) => {
  const currentTime = useMemo(() => {
    if (events.length === 0) return 0;
    return Math.max(...events.map(e => e.timestamp));
  }, [events]);

  const windowStart = Math.max(0, currentTime - windowDuration);

  const eventsByDrum = useMemo(() => {
    const grouped: Record<string, DrumEvent[]> = {};
    DRUM_ROWS.forEach(drum => {
      grouped[drum] = [];
    });

    events.forEach(event => {
      if (event.timestamp >= windowStart && grouped[event.drum_class]) {
        grouped[event.drum_class].push(event);
      }
    });

    return grouped;
  }, [events, windowStart]);

  const getEventPosition = (timestamp: number): number => {
    const relativeTime = timestamp - windowStart;
    return (relativeTime / windowDuration) * 100;
  };

  const getEventWidth = (velocity: number): number => {
    return Math.max(1, (velocity / 127) * 3);
  };

  return (
    <div className="midi-preview">
      <div className="midi-preview-header">
        <h4>MIDI Preview</h4>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          {events.length} events
        </span>
      </div>

      <div className="midi-grid">
        {DRUM_ROWS.map(drum => (
          <div key={drum} className="midi-row">
            <div className="midi-label">{drum}</div>
            <div className="midi-track">
              {eventsByDrum[drum].map((event, idx) => (
                <div
                  key={`${event.timestamp}-${idx}`}
                  className={`midi-note ${event.drum_class}`}
                  style={{
                    left: `${getEventPosition(event.timestamp)}%`,
                    width: `${getEventWidth(event.velocity)}%`,
                    opacity: 0.6 + (event.confidence * 0.4),
                  }}
                  title={`${event.drum_class} - vel: ${event.velocity}, conf: ${(event.confidence * 100).toFixed(0)}%`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
