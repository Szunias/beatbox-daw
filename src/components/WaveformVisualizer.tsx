import React, { useRef, useEffect, useCallback } from 'react';

interface WaveformVisualizerProps {
  audioLevel: number;
  isRecording: boolean;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  audioLevel,
  isRecording,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<number[]>([]);
  const animationRef = useRef<number>();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Update history
    historyRef.current.push(audioLevel);
    if (historyRef.current.length > canvas.width / 2) {
      historyRef.current.shift();
    }

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let y = 0; y < canvas.height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw waveform
    const history = historyRef.current;
    const midY = canvas.height / 2;
    const maxHeight = canvas.height * 0.4;

    ctx.beginPath();
    ctx.strokeStyle = isRecording ? '#e94560' : '#4ade80';
    ctx.lineWidth = 2;

    history.forEach((level, i) => {
      const x = i * 2;
      const height = level * maxHeight;

      if (i === 0) {
        ctx.moveTo(x, midY);
      }

      // Draw symmetric waveform
      ctx.lineTo(x, midY - height);
    });

    ctx.stroke();

    // Draw mirror
    ctx.beginPath();
    ctx.strokeStyle = isRecording ? 'rgba(233, 69, 96, 0.5)' : 'rgba(74, 222, 128, 0.5)';

    history.forEach((level, i) => {
      const x = i * 2;
      const height = level * maxHeight;

      if (i === 0) {
        ctx.moveTo(x, midY);
      }

      ctx.lineTo(x, midY + height);
    });

    ctx.stroke();

    // Draw center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(canvas.width, midY);
    ctx.stroke();

    // Draw current level indicator
    const currentX = history.length * 2;
    ctx.fillStyle = isRecording ? '#e94560' : '#4ade80';
    ctx.beginPath();
    ctx.arc(currentX, midY, 4, 0, Math.PI * 2);
    ctx.fill();

    animationRef.current = requestAnimationFrame(draw);
  }, [audioLevel, isRecording]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size
    const updateSize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);

    // Start animation
    animationRef.current = requestAnimationFrame(draw);

    return () => {
      window.removeEventListener('resize', updateSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [draw]);

  return (
    <canvas ref={canvasRef} className="waveform-canvas" />
  );
};
