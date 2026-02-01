import { useState, useEffect, useCallback, useRef } from 'react';
import { demoPatterns, DemoPattern, generatePatternEvents } from '../data/demoPatterns';

export interface DrumEvent {
  drum_class: string;
  confidence: number;
  midi_note: number;
  velocity: number;
  timestamp: number;
}

export interface EngineStatus {
  state: 'stopped' | 'running' | 'paused';
  events_detected: number;
  midi_connected: boolean;
  uptime: number;
  transport?: {
    state: string;
    current_tick: number;
    bpm: number;
  };
}

export interface AudioDevice {
  id: number;
  name: string;
  channels: number;
  sample_rate: number;
  type?: 'input' | 'output' | 'both';
}

export interface TransportPosition {
  tick: number;
  state: string;
  bpm: number;
}

export interface WebSocketMessage {
  type: string;
  data: unknown;
}

export interface RecordingCompletedData {
  trackId: string;
  startTick: number;
  durationSeconds: number;
  audioFilePath?: string;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  isDemoMode: boolean;
  status: EngineStatus | null;
  audioLevel: number;
  recentEvents: DrumEvent[];
  devices: AudioDevice[];
  transportPosition: TransportPosition | null;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (type: string, data?: Record<string, unknown>) => void;
  startEngine: () => void;
  stopEngine: () => void;
  startRecording: () => void;
  stopRecording: () => void;
  exportMidi: (filename: string, bpm?: number) => void;
  // Transport controls
  transportPlay: () => void;
  transportPause: () => void;
  transportStop: () => void;
  transportRecord: () => void;
  transportSeek: (tick: number) => void;
  setBpm: (bpm: number) => void;
  setLoop: (enabled: boolean, startTick?: number, endTick?: number) => void;
  setClick: (enabled: boolean) => void;
  // Audio devices
  listDevices: () => void;
  setAudioDevice: (deviceId: number, type: 'input' | 'output') => void;
  // Callbacks for external listeners
  onTransportPosition: (callback: (position: TransportPosition) => void) => () => void;
  onRecordingCompleted: (callback: (data: RecordingCompletedData) => void) => () => void;
  // Demo mode
  playDemoPattern: (patternIndex?: number) => void;
  stopDemoPattern: () => void;
  availablePatterns: DemoPattern[];
}

const MAX_EVENTS = 100;
const DEMO_MODE_TIMEOUT = 3000;

// Exponential backoff configuration
const RECONNECT_MIN_DELAY = 1000;
const RECONNECT_MAX_DELAY = 30000;
const RECONNECT_MULTIPLIER = 2;
const MAX_COMMAND_QUEUE_SIZE = 100;

interface QueuedCommand {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
}

export function useWebSocket(url: string = import.meta.env.VITE_WS_URL || 'ws://localhost:8765'): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recentEvents, setRecentEvents] = useState<DrumEvent[]>([]);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [transportPosition, setTransportPosition] = useState<TransportPosition | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const demoModeTimeoutRef = useRef<number | null>(null);
  const demoPlaybackRef = useRef<number | null>(null);
  const demoLevelRef = useRef<number | null>(null);
  const demoEventIndexRef = useRef(0);
  const transportPositionCallbacksRef = useRef<Set<(position: TransportPosition) => void>>(new Set());
  const commandQueueRef = useRef<QueuedCommand[]>([]);
  const transportPositionRef = useRef<TransportPosition | null>(null);
  const recordingStartTickRef = useRef<number>(0);
  const recordingTrackIdRef = useRef<string | null>(null);
  const recordingCompletedCallbacksRef = useRef<Set<(data: RecordingCompletedData) => void>>(new Set());

  // Calculate exponential backoff delay
  const getReconnectDelay = useCallback(() => {
    const delay = RECONNECT_MIN_DELAY * Math.pow(RECONNECT_MULTIPLIER, reconnectAttemptsRef.current);
    return Math.min(delay, RECONNECT_MAX_DELAY);
  }, []);

  // Flush queued commands when connection is established
  const flushCommandQueue = useCallback((ws: WebSocket) => {
    const queue = commandQueueRef.current;
    if (queue.length === 0) return;

    // Filter out stale commands (older than 30 seconds)
    const now = Date.now();
    const validCommands = queue.filter(cmd => now - cmd.timestamp < 30000);

    validCommands.forEach(cmd => {
      try {
        ws.send(JSON.stringify({ type: cmd.type, data: cmd.data }));
      } catch {
        // Ignore send errors during flush
      }
    });

    // Clear the queue
    commandQueueRef.current = [];
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Clear any existing demo mode timeout
    if (demoModeTimeoutRef.current) {
      clearTimeout(demoModeTimeoutRef.current);
    }

    // Set demo mode timeout - if no connection after 3 seconds, enable demo mode
    demoModeTimeoutRef.current = window.setTimeout(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.log('Backend unavailable, switching to demo mode');
        setIsDemoMode(true);
      }
    }, DEMO_MODE_TIMEOUT);

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        setIsDemoMode(false);

        // Reset reconnect attempts on successful connection
        reconnectAttemptsRef.current = 0;

        // Clear demo mode timeout on successful connection
        if (demoModeTimeoutRef.current) {
          clearTimeout(demoModeTimeoutRef.current);
          demoModeTimeoutRef.current = null;
        }

        // Flush any queued commands
        flushCommandQueue(ws);

        // Request initial status
        ws.send(JSON.stringify({ type: 'status' }));
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;

        // Auto-reconnect with exponential backoff
        if (reconnectTimeoutRef.current === null) {
          const delay = getReconnectDelay();
          reconnectAttemptsRef.current++;

          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connect();
          }, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      };

      wsRef.current = ws;
    } catch {
      // Connection failed, will retry with backoff
      const delay = getReconnectDelay();
      reconnectAttemptsRef.current++;

      if (reconnectTimeoutRef.current === null) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          reconnectTimeoutRef.current = null;
          connect();
        }, delay);
      }
    }
  }, [url, flushCommandQueue, getReconnectDelay]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current !== null) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Reset reconnect state
    reconnectAttemptsRef.current = 0;
    // Clear command queue on intentional disconnect
    commandQueueRef.current = [];

    setIsConnected(false);
  }, []);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'status':
        setStatus(message.data as EngineStatus);
        break;

      case 'audio_level':
        setAudioLevel((message.data as { level: number }).level);
        break;

      case 'drum_event':
        const event = message.data as DrumEvent;
        setRecentEvents(prev => {
          const updated = [event, ...prev];
          return updated.slice(0, MAX_EVENTS);
        });
        break;

      case 'start_response':
      case 'stop_response':
      case 'pause_response':
      case 'resume_response':
        // Request updated status after state change
        sendMessage('status');
        break;

      case 'recording_started':
        // Legacy recording started (MIDI events)
        break;

      case 'recording_stopped':
        // Legacy recording stopped (MIDI events)
        break;

      case 'track_recording_started': {
        const startData = message.data as { track_id: string; start_time: number };
        // Store the start tick - use the ref to get current transport position
        const currentTick = transportPositionRef.current?.tick || 0;
        recordingStartTickRef.current = currentTick;
        recordingTrackIdRef.current = startData.track_id;
        break;
      }

      case 'track_recording_stopped': {
        const stopData = message.data as {
          track_id: string;
          duration: number;
          sample_rate: number;
          num_samples: number;
          start_time: number;
        };

        // Create recording completed data
        const recordingData: RecordingCompletedData = {
          trackId: stopData.track_id || recordingTrackIdRef.current || '',
          startTick: recordingStartTickRef.current,
          durationSeconds: stopData.duration,
          // Audio file path would be set by the backend when exporting
          audioFilePath: undefined,
        };

        // Notify all registered callbacks
        recordingCompletedCallbacksRef.current.forEach(callback => callback(recordingData));

        // Reset recording state
        recordingStartTickRef.current = 0;
        recordingTrackIdRef.current = null;
        break;
      }

      case 'track_recording_auto_stopped': {
        const autoStopData = message.data as {
          track_id: string;
          duration: number;
          reason: string;
        };

        // Handle auto-stop (e.g., max duration reached) same as regular stop
        const recordingData: RecordingCompletedData = {
          trackId: autoStopData.track_id || recordingTrackIdRef.current || '',
          startTick: recordingStartTickRef.current,
          durationSeconds: autoStopData.duration,
        };

        recordingCompletedCallbacksRef.current.forEach(callback => callback(recordingData));

        recordingStartTickRef.current = 0;
        recordingTrackIdRef.current = null;
        break;
      }

      case 'export_response':
        const exportData = message.data as { success: boolean; filename: string };
        if (exportData.success) {
          console.log(`MIDI exported to: ${exportData.filename}`);
        }
        break;

      case 'pong':
        // Heartbeat response
        break;

      // Transport messages
      case 'transport_position': {
        const position = message.data as TransportPosition;
        setTransportPosition(position);
        transportPositionRef.current = position;
        // Notify all registered callbacks
        transportPositionCallbacksRef.current.forEach(callback => callback(position));
        break;
      }

      case 'transport_state': {
        const stateData = message.data as { state: string; tick: number };
        setTransportPosition(prev => prev ? { ...prev, state: stateData.state, tick: stateData.tick } : null);
        break;
      }

      case 'transport_play_response':
      case 'transport_pause_response':
      case 'transport_stop_response':
      case 'transport_record_response':
      case 'transport_seek_response':
      case 'set_bpm_response':
      case 'set_loop_response':
      case 'set_click_response':
        // Transport responses handled
        break;

      // Device messages
      case 'devices': {
        const deviceData = message.data as { devices: AudioDevice[] };
        setDevices(deviceData.devices);
        break;
      }

      case 'set_audio_device_response':
        console.log('Audio device changed:', message.data);
        break;

      case 'set_track_armed_response':
        // Track armed state synced with backend
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, []);

  const sendMessage = useCallback((type: string, data: Record<string, unknown> = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, data }));
    } else {
      // Queue command for later when not connected
      // Skip queueing ping messages as they're not meaningful when disconnected
      if (type !== 'ping') {
        const queue = commandQueueRef.current;
        // Limit queue size to prevent memory issues
        if (queue.length < MAX_COMMAND_QUEUE_SIZE) {
          queue.push({
            type,
            data,
            timestamp: Date.now()
          });
        }
      }
    }
  }, []);

  const startEngine = useCallback(() => {
    sendMessage('start');
  }, [sendMessage]);

  const stopEngine = useCallback(() => {
    sendMessage('stop');
  }, [sendMessage]);

  const startRecording = useCallback(() => {
    sendMessage('start_recording');
  }, [sendMessage]);

  const stopRecording = useCallback(() => {
    sendMessage('stop_recording');
  }, [sendMessage]);

  const exportMidi = useCallback((filename: string, bpm: number = 120) => {
    sendMessage('export_midi', { filename, bpm });
  }, [sendMessage]);

  // === Transport Controls ===
  const transportPlay = useCallback(() => {
    sendMessage('transport_play');
  }, [sendMessage]);

  const transportPause = useCallback(() => {
    sendMessage('transport_pause');
  }, [sendMessage]);

  const transportStop = useCallback(() => {
    sendMessage('transport_stop');
  }, [sendMessage]);

  const transportRecord = useCallback(() => {
    sendMessage('transport_record');
  }, [sendMessage]);

  const transportSeek = useCallback((tick: number) => {
    sendMessage('transport_seek', { tick });
  }, [sendMessage]);

  const setBpm = useCallback((bpm: number) => {
    sendMessage('set_bpm', { bpm });
  }, [sendMessage]);

  const setLoop = useCallback((enabled: boolean, startTick?: number, endTick?: number) => {
    sendMessage('set_loop', { enabled, start_tick: startTick, end_tick: endTick });
  }, [sendMessage]);

  const setClick = useCallback((enabled: boolean) => {
    sendMessage('set_click', { enabled });
  }, [sendMessage]);

  // === Audio Devices ===
  const listDevices = useCallback(() => {
    sendMessage('list_devices');
  }, [sendMessage]);

  const setAudioDevice = useCallback((deviceId: number, type: 'input' | 'output') => {
    sendMessage('set_audio_device', { device_id: deviceId, type });
  }, [sendMessage]);

  // === Transport Position Callback ===
  const onTransportPosition = useCallback((callback: (position: TransportPosition) => void) => {
    transportPositionCallbacksRef.current.add(callback);
    // Return unsubscribe function
    return () => {
      transportPositionCallbacksRef.current.delete(callback);
    };
  }, []);

  // === Recording Completed Callback ===
  const onRecordingCompleted = useCallback((callback: (data: RecordingCompletedData) => void) => {
    recordingCompletedCallbacksRef.current.add(callback);
    // Return unsubscribe function
    return () => {
      recordingCompletedCallbacksRef.current.delete(callback);
    };
  }, []);

  // === Demo Mode Functions ===
  const stopDemoPattern = useCallback(() => {
    if (demoPlaybackRef.current) {
      clearInterval(demoPlaybackRef.current);
      demoPlaybackRef.current = null;
    }
    if (demoLevelRef.current) {
      cancelAnimationFrame(demoLevelRef.current);
      demoLevelRef.current = null;
    }
    setAudioLevel(0);
    demoEventIndexRef.current = 0;
  }, []);

  const playDemoPattern = useCallback((patternIndex?: number) => {
    if (!isDemoMode) return;

    stopDemoPattern();

    const pattern = patternIndex !== undefined
      ? demoPatterns[patternIndex]
      : demoPatterns[Math.floor(Math.random() * demoPatterns.length)];

    const startTime = Date.now();
    demoEventIndexRef.current = 0;

    // Schedule events
    const playbackLoop = () => {
      const elapsed = (Date.now() - startTime) % pattern.duration;

      // Find and emit events that should play now
      while (demoEventIndexRef.current < pattern.events.length) {
        const event = pattern.events[demoEventIndexRef.current];
        if (event.timestamp <= elapsed) {
          const newEvent: DrumEvent = {
            ...event,
            timestamp: Date.now(),
          };
          setRecentEvents((prev) => [newEvent, ...prev.slice(0, MAX_EVENTS - 1)]);

          // Simulate audio level spike
          const level = 0.5 + (event.velocity / 127) * 0.5;
          setAudioLevel(level);

          demoEventIndexRef.current++;
        } else {
          break;
        }
      }

      // Reset for loop
      if (elapsed < 50 && demoEventIndexRef.current >= pattern.events.length) {
        demoEventIndexRef.current = 0;
      }
    };

    demoPlaybackRef.current = window.setInterval(playbackLoop, 16);

    // Animate audio level decay
    const animateLevel = () => {
      setAudioLevel((prev) => Math.max(prev * 0.92, 0.02));
      demoLevelRef.current = requestAnimationFrame(animateLevel);
    };
    demoLevelRef.current = requestAnimationFrame(animateLevel);
  }, [isDemoMode, stopDemoPattern]);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
      stopDemoPattern();
      if (demoModeTimeoutRef.current) {
        clearTimeout(demoModeTimeoutRef.current);
      }
    };
  }, [connect, disconnect, stopDemoPattern]);

  // Heartbeat
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      sendMessage('ping');
    }, 30000);

    return () => clearInterval(interval);
  }, [isConnected, sendMessage]);

  return {
    isConnected,
    isDemoMode,
    status,
    audioLevel,
    recentEvents,
    devices,
    transportPosition,
    connect,
    disconnect,
    sendMessage,
    startEngine,
    stopEngine,
    startRecording,
    stopRecording,
    exportMidi,
    // Transport controls
    transportPlay,
    transportPause,
    transportStop,
    transportRecord,
    transportSeek,
    setBpm,
    setLoop,
    setClick,
    // Audio devices
    listDevices,
    setAudioDevice,
    // Callbacks
    onTransportPosition,
    onRecordingCompleted,
    // Demo mode
    playDemoPattern,
    stopDemoPattern,
    availablePatterns: demoPatterns,
  };
}
