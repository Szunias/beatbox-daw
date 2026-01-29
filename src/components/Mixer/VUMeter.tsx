/**
 * VUMeter Component
 * Visual audio level meter for mixer channels
 */

import React, { useEffect, useRef } from 'react';

interface VUMeterProps {
  level: number; // 0-1
  width?: number;
  height?: number;
  orientation?: 'vertical' | 'horizontal';
  showPeak?: boolean;
  peakHoldTime?: number;
}

export const VUMeter: React.FC<VUMeterProps> = ({
  level,
  width = 12,
  height = 150,
  orientation = 'vertical',
  showPeak = true,
  peakHoldTime = 1500,
}) => {
  const peakRef = useRef<number>(0);
  const peakTimeRef = useRef<number>(0);

  // Update peak
  useEffect(() => {
    if (level > peakRef.current) {
      peakRef.current = level;
      peakTimeRef.current = Date.now();
    } else if (Date.now() - peakTimeRef.current > peakHoldTime) {
      peakRef.current = Math.max(level, peakRef.current * 0.95);
    }
  }, [level, peakHoldTime]);

  const isVertical = orientation === 'vertical';
  const meterSize = isVertical ? height : width;
  const levelSize = level * meterSize;
  const peakPosition = peakRef.current * meterSize;

  // Calculate gradient segments (green -> yellow -> red)
  const greenEnd = meterSize * 0.6;
  const yellowEnd = meterSize * 0.85;

  // Get color based on level
  const getColorAtPosition = (pos: number): string => {
    if (pos < greenEnd) return 'var(--success)';
    if (pos < yellowEnd) return 'var(--warning)';
    return 'var(--error)';
  };

  return (
    <div
      className="vu-meter"
      style={{
        width: isVertical ? width : height,
        height: isVertical ? height : width,
        backgroundColor: 'var(--bg-primary)',
        borderRadius: 3,
        overflow: 'hidden',
        position: 'relative',
        display: 'flex',
        flexDirection: isVertical ? 'column-reverse' : 'row',
      }}
    >
      {/* Level indicator */}
      <div
        style={{
          position: 'absolute',
          [isVertical ? 'bottom' : 'left']: 0,
          [isVertical ? 'left' : 'top']: 0,
          [isVertical ? 'width' : 'height']: '100%',
          [isVertical ? 'height' : 'width']: `${level * 100}%`,
          background: `linear-gradient(${isVertical ? 'to top' : 'to right'},
            var(--success) 0%,
            var(--success) 60%,
            var(--warning) 60%,
            var(--warning) 85%,
            var(--error) 85%,
            var(--error) 100%
          )`,
          transition: 'height 0.05s ease, width 0.05s ease',
        }}
      />

      {/* Peak indicator */}
      {showPeak && peakRef.current > 0.01 && (
        <div
          style={{
            position: 'absolute',
            [isVertical ? 'bottom' : 'left']: `${peakRef.current * 100}%`,
            [isVertical ? 'left' : 'top']: 0,
            [isVertical ? 'width' : 'height']: '100%',
            [isVertical ? 'height' : 'width']: 2,
            backgroundColor: getColorAtPosition(peakPosition),
            transform: isVertical ? 'translateY(50%)' : 'translateX(-50%)',
          }}
        />
      )}

      {/* Segment lines */}
      {[...Array(10)].map((_, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            [isVertical ? 'bottom' : 'left']: `${(i + 1) * 10}%`,
            [isVertical ? 'left' : 'top']: 0,
            [isVertical ? 'width' : 'height']: '100%',
            [isVertical ? 'height' : 'width']: 1,
            backgroundColor: 'rgba(0,0,0,0.3)',
          }}
        />
      ))}
    </div>
  );
};

export default VUMeter;
