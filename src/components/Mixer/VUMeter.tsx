/**
 * VUMeter Component
 * Visual audio level meter for mixer channels
 * Features modern gradient design with smooth animations and glow effects
 */

import React, { useEffect, useRef, useState } from 'react';
import { motion, useSpring, useTransform } from 'motion/react';
import { cn } from '../../lib/utils';

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
  const [peak, setPeak] = useState<number>(0);
  const peakTimeRef = useRef<number>(0);
  const isVertical = orientation === 'vertical';

  // Smooth spring animation for level
  const springLevel = useSpring(level, {
    stiffness: 300,
    damping: 30,
    mass: 0.5,
  });

  // Transform spring value to percentage
  const levelPercentage = useTransform(springLevel, [0, 1], [0, 100]);

  // Update spring when level changes
  useEffect(() => {
    springLevel.set(level);
  }, [level, springLevel]);

  // Update peak with decay
  useEffect(() => {
    if (level > peak) {
      setPeak(level);
      peakTimeRef.current = Date.now();
    } else if (Date.now() - peakTimeRef.current > peakHoldTime) {
      setPeak((prev) => Math.max(level, prev * 0.92));
    }
  }, [level, peak, peakHoldTime]);

  // Decay peak gradually
  useEffect(() => {
    const interval = setInterval(() => {
      if (Date.now() - peakTimeRef.current > peakHoldTime) {
        setPeak((prev) => Math.max(0, prev * 0.92));
      }
    }, 50);
    return () => clearInterval(interval);
  }, [peakHoldTime]);

  // Get color class based on level position
  const getPeakColor = (peakLevel: number): string => {
    if (peakLevel < 0.6) return 'bg-green-400';
    if (peakLevel < 0.85) return 'bg-amber-400';
    return 'bg-red-500';
  };

  // Get glow color for peak
  const getPeakGlow = (peakLevel: number): string => {
    if (peakLevel < 0.6) return 'shadow-green-400/60';
    if (peakLevel < 0.85) return 'shadow-amber-400/60';
    return 'shadow-red-500/60';
  };

  return (
    <div
      className={cn(
        'vu-meter',
        'relative overflow-hidden',
        'rounded-sm',
        // Modern glass background
        'bg-slate-900/80 backdrop-blur-sm',
        'border border-slate-700/30'
      )}
      style={{
        width: isVertical ? width : height,
        height: isVertical ? height : width,
      }}
    >
      {/* Background gradient overlay for depth */}
      <div
        className={cn(
          'absolute inset-0',
          'bg-gradient-to-t from-slate-950/50 via-transparent to-slate-800/30',
          'pointer-events-none'
        )}
      />

      {/* Level indicator with modern gradient */}
      <motion.div
        className={cn(
          'absolute',
          isVertical ? 'bottom-0 left-0 right-0' : 'top-0 left-0 bottom-0',
          'rounded-sm'
        )}
        style={{
          [isVertical ? 'height' : 'width']: useTransform(
            levelPercentage,
            (v) => `${v}%`
          ),
          // Modern gradient with glow
          background: isVertical
            ? `linear-gradient(to top,
                #22c55e 0%,
                #4ade80 25%,
                #4ade80 55%,
                #fbbf24 65%,
                #f59e0b 75%,
                #ef4444 85%,
                #dc2626 100%
              )`
            : `linear-gradient(to right,
                #22c55e 0%,
                #4ade80 25%,
                #4ade80 55%,
                #fbbf24 65%,
                #f59e0b 75%,
                #ef4444 85%,
                #dc2626 100%
              )`,
          boxShadow: level > 0.1
            ? `0 0 ${Math.min(level * 12, 8)}px ${level < 0.6 ? 'rgba(74, 222, 128, 0.5)' : level < 0.85 ? 'rgba(251, 191, 36, 0.5)' : 'rgba(239, 68, 68, 0.6)'}`
            : 'none',
        }}
      />

      {/* Peak indicator with glow */}
      {showPeak && peak > 0.02 && (
        <motion.div
          className={cn(
            'absolute z-10',
            isVertical ? 'left-0 right-0' : 'top-0 bottom-0',
            getPeakColor(peak),
            'shadow-lg',
            getPeakGlow(peak)
          )}
          style={{
            [isVertical ? 'bottom' : 'left']: `${peak * 100}%`,
            [isVertical ? 'height' : 'width']: 2,
            transform: isVertical ? 'translateY(50%)' : 'translateX(-50%)',
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.1 }}
        />
      )}

      {/* Segment lines for visual reference */}
      {[...Array(10)].map((_, i) => (
        <div
          key={i}
          className={cn(
            'absolute',
            isVertical ? 'left-0 right-0' : 'top-0 bottom-0',
            'bg-slate-900/40'
          )}
          style={{
            [isVertical ? 'bottom' : 'left']: `${(i + 1) * 10}%`,
            [isVertical ? 'height' : 'width']: 1,
          }}
        />
      ))}

      {/* Danger zone indicator (top 15%) */}
      <div
        className={cn(
          'absolute pointer-events-none',
          isVertical ? 'left-0 right-0 top-0' : 'top-0 bottom-0 right-0',
          'border-b border-red-500/30'
        )}
        style={{
          [isVertical ? 'height' : 'width']: '15%',
        }}
      />

      {/* Warning zone indicator (15-40% from top) */}
      <div
        className={cn(
          'absolute pointer-events-none',
          isVertical ? 'left-0 right-0' : 'top-0 bottom-0',
          'border-b border-amber-500/20'
        )}
        style={{
          [isVertical ? 'top' : 'right']: '15%',
          [isVertical ? 'height' : 'width']: '25%',
        }}
      />

      {/* Clip indicator when level hits max */}
      {level >= 0.98 && (
        <motion.div
          className={cn(
            'absolute top-0 left-0 right-0',
            'h-1 bg-red-500',
            'shadow-lg shadow-red-500/80'
          )}
          initial={{ opacity: 0 }}
          animate={{ opacity: [1, 0.5, 1] }}
          transition={{ duration: 0.3, repeat: Infinity }}
        />
      )}
    </div>
  );
};

export default VUMeter;
