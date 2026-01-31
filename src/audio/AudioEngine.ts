/**
 * AudioEngine
 * Web Audio API based audio playback engine for the DAW
 * Handles audio clip scheduling to start at correct timeline positions
 */

import { AudioClip, Track, TICKS_PER_BEAT, ticksToSeconds } from '../types/project';

// Configuration
const SCHEDULE_AHEAD_TIME = 0.1; // Schedule audio 100ms ahead
const SCHEDULER_INTERVAL = 25; // Run scheduler every 25ms

// Types
interface LoadedAudioClip {
  clipId: string;
  trackId: string;
  audioBuffer: AudioBuffer;
  startTick: number;
  duration: number; // in ticks
  muted: boolean;
}

interface ScheduledSource {
  source: AudioBufferSourceNode;
  gainNode: GainNode;
  pannerNode: StereoPannerNode;
  clipId: string;
  trackId: string;
  startTime: number;
  endTime: number;
}

interface TrackSettings {
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  gainNode: GainNode;
  pannerNode: StereoPannerNode;
}

type TransportState = 'stopped' | 'playing' | 'paused' | 'recording';

export class AudioEngine {
  private audioContext: AudioContext | null = null;
  private masterGainNode: GainNode | null = null;

  // Loaded audio data
  private loadedClips: Map<string, LoadedAudioClip> = new Map();

  // Track mixer settings
  private trackSettings: Map<string, TrackSettings> = new Map();

  // Currently scheduled/playing sources
  private scheduledSources: ScheduledSource[] = [];

  // Scheduler state
  private schedulerTimer: number | null = null;
  private isPlaying: boolean = false;
  private playbackStartTime: number = 0; // AudioContext time when playback started
  private playbackStartTick: number = 0; // Transport tick when playback started

  // Project settings
  private bpm: number = 120;

  // Solo tracking
  private hasSoloedTrack: boolean = false;
  private soloedTrackIds: Set<string> = new Set();

  // Callbacks
  private onClipStarted?: (clipId: string, trackId: string) => void;
  private onClipEnded?: (clipId: string, trackId: string) => void;

  /**
   * Initialize the audio engine
   */
  async init(): Promise<boolean> {
    try {
      this.audioContext = new AudioContext();

      // Create master gain node
      this.masterGainNode = this.audioContext.createGain();
      this.masterGainNode.gain.value = 1.0;
      this.masterGainNode.connect(this.audioContext.destination);

      return true;
    } catch (error) {
      console.error('Failed to initialize AudioEngine:', error);
      return false;
    }
  }

  /**
   * Resume audio context (required after user gesture on some browsers)
   */
  async resume(): Promise<void> {
    if (this.audioContext?.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  /**
   * Get the audio context (for external use like analyzers)
   */
  getAudioContext(): AudioContext | null {
    return this.audioContext;
  }

  /**
   * Set the master volume
   */
  setMasterVolume(volume: number): void {
    if (this.masterGainNode) {
      this.masterGainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  /**
   * Set the BPM for time calculations
   */
  setBpm(bpm: number): void {
    this.bpm = Math.max(20, Math.min(300, bpm));
  }

  /**
   * Load an audio file as a clip
   */
  async loadAudioClip(
    clipId: string,
    trackId: string,
    audioUrl: string,
    startTick: number,
    durationTicks: number,
    muted: boolean = false
  ): Promise<boolean> {
    if (!this.audioContext) {
      console.error('AudioEngine not initialized');
      return false;
    }

    try {
      const response = await fetch(audioUrl);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

      this.loadedClips.set(clipId, {
        clipId,
        trackId,
        audioBuffer,
        startTick,
        duration: durationTicks,
        muted,
      });

      return true;
    } catch (error) {
      console.error(`Failed to load audio clip ${clipId}:`, error);
      return false;
    }
  }

  /**
   * Load audio from an ArrayBuffer directly
   */
  async loadAudioBuffer(
    clipId: string,
    trackId: string,
    arrayBuffer: ArrayBuffer,
    startTick: number,
    durationTicks: number,
    muted: boolean = false
  ): Promise<boolean> {
    if (!this.audioContext) {
      console.error('AudioEngine not initialized');
      return false;
    }

    try {
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

      this.loadedClips.set(clipId, {
        clipId,
        trackId,
        audioBuffer,
        startTick,
        duration: durationTicks,
        muted,
      });

      return true;
    } catch (error) {
      console.error(`Failed to load audio buffer ${clipId}:`, error);
      return false;
    }
  }

  /**
   * Load audio from an AudioBuffer directly
   */
  loadAudioBufferDirect(
    clipId: string,
    trackId: string,
    audioBuffer: AudioBuffer,
    startTick: number,
    durationTicks: number,
    muted: boolean = false
  ): void {
    this.loadedClips.set(clipId, {
      clipId,
      trackId,
      audioBuffer,
      startTick,
      duration: durationTicks,
      muted,
    });
  }

  /**
   * Unload an audio clip
   */
  unloadAudioClip(clipId: string): void {
    this.loadedClips.delete(clipId);

    // Stop any scheduled playback for this clip
    this.scheduledSources = this.scheduledSources.filter((s) => {
      if (s.clipId === clipId) {
        try {
          s.source.stop();
        } catch (e) {
          // Ignore errors if already stopped
        }
        return false;
      }
      return true;
    });
  }

  /**
   * Clear all loaded clips
   */
  clearAllClips(): void {
    this.stopAllSources();
    this.loadedClips.clear();
  }

  /**
   * Update track mixer settings
   */
  updateTrackSettings(
    trackId: string,
    volume?: number,
    pan?: number,
    muted?: boolean,
    solo?: boolean
  ): void {
    if (!this.audioContext || !this.masterGainNode) return;

    let settings = this.trackSettings.get(trackId);

    if (!settings) {
      // Create new track nodes
      const gainNode = this.audioContext.createGain();
      const pannerNode = this.audioContext.createStereoPanner();

      // Connect: panner -> gain -> master
      pannerNode.connect(gainNode);
      gainNode.connect(this.masterGainNode);

      settings = {
        volume: 1.0,
        pan: 0,
        muted: false,
        solo: false,
        gainNode,
        pannerNode,
      };
      this.trackSettings.set(trackId, settings);
    }

    // Update values
    if (volume !== undefined) {
      settings.volume = Math.max(0, Math.min(1, volume));
    }
    if (pan !== undefined) {
      settings.pan = Math.max(-1, Math.min(1, pan));
      settings.pannerNode.pan.value = settings.pan;
    }
    if (muted !== undefined) {
      settings.muted = muted;
    }
    if (solo !== undefined) {
      const wasSoloed = settings.solo;
      settings.solo = solo;

      // Update solo tracking
      if (solo && !wasSoloed) {
        this.soloedTrackIds.add(trackId);
      } else if (!solo && wasSoloed) {
        this.soloedTrackIds.delete(trackId);
      }
      this.hasSoloedTrack = this.soloedTrackIds.size > 0;
    }

    // Apply effective gain (considering mute and solo)
    this.applyTrackGain(trackId);
  }

  /**
   * Apply the effective gain to a track (considering mute/solo)
   */
  private applyTrackGain(trackId: string): void {
    const settings = this.trackSettings.get(trackId);
    if (!settings) return;

    let effectiveGain = settings.volume;

    // Check mute
    if (settings.muted) {
      effectiveGain = 0;
    }

    // Check solo - if any track is soloed, mute non-soloed tracks
    if (this.hasSoloedTrack && !settings.solo) {
      effectiveGain = 0;
    }

    settings.gainNode.gain.value = effectiveGain;
  }

  /**
   * Re-apply gains to all tracks (after solo state changes)
   */
  private reapplyAllTrackGains(): void {
    for (const trackId of this.trackSettings.keys()) {
      this.applyTrackGain(trackId);
    }
  }

  /**
   * Get or create track output nodes
   */
  private getTrackOutput(trackId: string): { gainNode: GainNode; pannerNode: StereoPannerNode } | null {
    if (!this.audioContext || !this.masterGainNode) return null;

    let settings = this.trackSettings.get(trackId);

    if (!settings) {
      // Create default settings
      this.updateTrackSettings(trackId);
      settings = this.trackSettings.get(trackId);
    }

    if (!settings) return null;

    return {
      gainNode: settings.gainNode,
      pannerNode: settings.pannerNode,
    };
  }

  /**
   * Start playback at a given tick position
   */
  startPlayback(startTick: number): void {
    if (!this.audioContext || this.isPlaying) return;

    this.resume();

    this.isPlaying = true;
    this.playbackStartTime = this.audioContext.currentTime;
    this.playbackStartTick = startTick;

    // Start the scheduler
    this.startScheduler();
  }

  /**
   * Stop playback
   */
  stopPlayback(): void {
    this.isPlaying = false;
    this.stopScheduler();
    this.stopAllSources();
  }

  /**
   * Pause playback (maintains position)
   */
  pausePlayback(): void {
    this.isPlaying = false;
    this.stopScheduler();
    this.stopAllSources();
  }

  /**
   * Seek to a new position (if playing, restart from there)
   */
  seekTo(tick: number): void {
    const wasPlaying = this.isPlaying;

    // Stop current playback
    this.stopPlayback();

    if (wasPlaying) {
      // Restart from new position
      this.startPlayback(tick);
    }
  }

  /**
   * Handle transport state changes
   */
  handleTransportStateChange(
    state: TransportState,
    currentTick: number,
    bpm: number
  ): void {
    this.bpm = bpm;

    switch (state) {
      case 'playing':
      case 'recording':
        if (!this.isPlaying) {
          this.startPlayback(currentTick);
        }
        break;

      case 'paused':
        this.pausePlayback();
        break;

      case 'stopped':
        this.stopPlayback();
        break;
    }
  }

  /**
   * Start the scheduling loop
   */
  private startScheduler(): void {
    if (this.schedulerTimer !== null) return;

    const scheduler = () => {
      this.scheduleAudioClips();
      this.schedulerTimer = window.setTimeout(scheduler, SCHEDULER_INTERVAL);
    };

    scheduler();
  }

  /**
   * Stop the scheduling loop
   */
  private stopScheduler(): void {
    if (this.schedulerTimer !== null) {
      clearTimeout(this.schedulerTimer);
      this.schedulerTimer = null;
    }
  }

  /**
   * Get the current playback tick based on audio context time
   */
  getCurrentTick(): number {
    if (!this.audioContext || !this.isPlaying) {
      return this.playbackStartTick;
    }

    const elapsedTime = this.audioContext.currentTime - this.playbackStartTime;
    const elapsedTicks = this.secondsToTicks(elapsedTime);

    return this.playbackStartTick + elapsedTicks;
  }

  /**
   * Convert ticks to seconds
   */
  private ticksToSeconds(ticks: number): number {
    return (ticks / TICKS_PER_BEAT) * (60 / this.bpm);
  }

  /**
   * Convert seconds to ticks
   */
  private secondsToTicks(seconds: number): number {
    return (seconds / 60) * this.bpm * TICKS_PER_BEAT;
  }

  /**
   * Schedule audio clips for playback
   * This is the core scheduling logic that ensures clips start at correct timeline positions
   */
  private scheduleAudioClips(): void {
    if (!this.audioContext || !this.isPlaying) return;

    const currentTime = this.audioContext.currentTime;
    const scheduleEndTime = currentTime + SCHEDULE_AHEAD_TIME;

    // Calculate tick range for scheduling
    const currentTick = this.getCurrentTick();
    const scheduleEndTick = currentTick + this.secondsToTicks(SCHEDULE_AHEAD_TIME);

    // Check each loaded clip
    for (const clip of this.loadedClips.values()) {
      // Skip muted clips
      if (clip.muted) continue;

      // Check if clip should start within the schedule window
      const clipStartTick = clip.startTick;
      const clipEndTick = clip.startTick + clip.duration;

      // Skip clips that have already ended
      if (clipEndTick <= currentTick) continue;

      // Check if this clip is already scheduled
      const alreadyScheduled = this.scheduledSources.some(
        (s) => s.clipId === clip.clipId && s.endTime > currentTime
      );
      if (alreadyScheduled) continue;

      // Check if clip should be scheduled
      // Case 1: Clip starts within schedule window
      // Case 2: Clip is currently playing (started before current tick but hasn't ended)
      const shouldSchedule =
        (clipStartTick >= currentTick && clipStartTick < scheduleEndTick) ||
        (clipStartTick < currentTick && clipEndTick > currentTick);

      if (!shouldSchedule) continue;

      // Schedule this clip
      this.scheduleClip(clip, currentTime, currentTick);
    }

    // Clean up finished sources
    this.cleanupFinishedSources();
  }

  /**
   * Schedule a single clip for playback
   */
  private scheduleClip(
    clip: LoadedAudioClip,
    currentAudioTime: number,
    currentTick: number
  ): void {
    if (!this.audioContext) return;

    const trackOutput = this.getTrackOutput(clip.trackId);
    if (!trackOutput) return;

    // Create source node
    const source = this.audioContext.createBufferSource();
    source.buffer = clip.audioBuffer;

    // Create clip-level gain node (for clip volume - if needed in future)
    const clipGain = this.audioContext.createGain();
    clipGain.gain.value = 1.0;

    // Create clip-level panner (for clip pan - if needed in future)
    const clipPanner = this.audioContext.createStereoPanner();
    clipPanner.pan.value = 0;

    // Connect: source -> clipGain -> clipPanner -> trackPanner -> trackGain -> master
    source.connect(clipGain);
    clipGain.connect(clipPanner);
    clipPanner.connect(trackOutput.pannerNode);

    // Calculate when to start playback
    let startTime: number;
    let offset = 0;

    if (clip.startTick >= currentTick) {
      // Clip starts in the future
      const ticksUntilStart = clip.startTick - currentTick;
      const secondsUntilStart = this.ticksToSeconds(ticksUntilStart);
      startTime = currentAudioTime + secondsUntilStart;
    } else {
      // Clip has already started - play from the middle
      const ticksIntoClip = currentTick - clip.startTick;
      const secondsIntoClip = this.ticksToSeconds(ticksIntoClip);
      offset = secondsIntoClip;
      startTime = currentAudioTime;
    }

    // Calculate clip duration in seconds
    const clipDurationSeconds = this.ticksToSeconds(clip.duration);
    const remainingDuration = clipDurationSeconds - offset;

    // Don't schedule if there's nothing left to play
    if (remainingDuration <= 0) return;

    // Calculate end time
    const endTime = startTime + remainingDuration;

    // Start playback
    source.start(startTime, offset, remainingDuration);

    // Track this source
    const scheduledSource: ScheduledSource = {
      source,
      gainNode: clipGain,
      pannerNode: clipPanner,
      clipId: clip.clipId,
      trackId: clip.trackId,
      startTime,
      endTime,
    };
    this.scheduledSources.push(scheduledSource);

    // Set up ended callback
    source.onended = () => {
      this.handleSourceEnded(scheduledSource);
    };

    // Fire clip started callback
    if (this.onClipStarted) {
      // Use setTimeout to fire at the actual start time
      const delay = (startTime - currentAudioTime) * 1000;
      if (delay > 0) {
        setTimeout(() => {
          this.onClipStarted?.(clip.clipId, clip.trackId);
        }, delay);
      } else {
        this.onClipStarted(clip.clipId, clip.trackId);
      }
    }
  }

  /**
   * Handle when an audio source ends
   */
  private handleSourceEnded(scheduledSource: ScheduledSource): void {
    // Remove from scheduled sources
    const index = this.scheduledSources.indexOf(scheduledSource);
    if (index !== -1) {
      this.scheduledSources.splice(index, 1);
    }

    // Fire callback
    if (this.onClipEnded) {
      this.onClipEnded(scheduledSource.clipId, scheduledSource.trackId);
    }
  }

  /**
   * Clean up finished audio sources
   */
  private cleanupFinishedSources(): void {
    if (!this.audioContext) return;

    const currentTime = this.audioContext.currentTime;

    this.scheduledSources = this.scheduledSources.filter((s) => {
      if (s.endTime < currentTime) {
        // Source has finished, try to stop it (may already be stopped)
        try {
          s.source.stop();
        } catch (e) {
          // Ignore - already stopped
        }
        return false;
      }
      return true;
    });
  }

  /**
   * Stop all currently playing sources
   */
  private stopAllSources(): void {
    for (const s of this.scheduledSources) {
      try {
        s.source.stop();
      } catch (e) {
        // Ignore errors if already stopped
      }
    }
    this.scheduledSources = [];
  }

  /**
   * Set callback for clip started events
   */
  setOnClipStarted(callback: (clipId: string, trackId: string) => void): void {
    this.onClipStarted = callback;
  }

  /**
   * Set callback for clip ended events
   */
  setOnClipEnded(callback: (clipId: string, trackId: string) => void): void {
    this.onClipEnded = callback;
  }

  /**
   * Get list of currently playing clip IDs
   */
  getActiveClips(): string[] {
    if (!this.audioContext) return [];

    const currentTime = this.audioContext.currentTime;
    return this.scheduledSources
      .filter((s) => s.startTime <= currentTime && s.endTime > currentTime)
      .map((s) => s.clipId);
  }

  /**
   * Check if a specific clip is currently playing
   */
  isClipPlaying(clipId: string): boolean {
    if (!this.audioContext) return false;

    const currentTime = this.audioContext.currentTime;
    return this.scheduledSources.some(
      (s) => s.clipId === clipId && s.startTime <= currentTime && s.endTime > currentTime
    );
  }

  /**
   * Dispose of the audio engine
   */
  dispose(): void {
    this.stopPlayback();
    this.clearAllClips();

    // Disconnect all track nodes
    for (const settings of this.trackSettings.values()) {
      settings.gainNode.disconnect();
      settings.pannerNode.disconnect();
    }
    this.trackSettings.clear();

    // Disconnect master
    if (this.masterGainNode) {
      this.masterGainNode.disconnect();
      this.masterGainNode = null;
    }

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  /**
   * Sync with project clips from the project store
   * This loads all audio clips from the project tracks
   */
  async syncWithProject(
    tracks: Track[],
    bpm: number
  ): Promise<void> {
    this.bpm = bpm;

    // Update track settings
    for (const track of tracks) {
      this.updateTrackSettings(
        track.id,
        track.volume,
        track.pan,
        track.muted,
        track.solo
      );

      // Update solo tracking
      if (track.solo) {
        this.soloedTrackIds.add(track.id);
      } else {
        this.soloedTrackIds.delete(track.id);
      }
    }

    this.hasSoloedTrack = this.soloedTrackIds.size > 0;
    this.reapplyAllTrackGains();

    // Note: Audio clip loading from URLs would be done separately
    // as it requires fetching the audio files
  }

  /**
   * Update mute state for a specific clip
   */
  setClipMuted(clipId: string, muted: boolean): void {
    const clip = this.loadedClips.get(clipId);
    if (clip) {
      clip.muted = muted;
    }
  }

  /**
   * Check if the engine is currently playing
   */
  isCurrentlyPlaying(): boolean {
    return this.isPlaying;
  }
}

// Singleton instance
let audioEngineInstance: AudioEngine | null = null;

/**
 * Get the singleton AudioEngine instance
 */
export function getAudioEngine(): AudioEngine {
  if (!audioEngineInstance) {
    audioEngineInstance = new AudioEngine();
  }
  return audioEngineInstance;
}

/**
 * Initialize the singleton AudioEngine
 */
export async function initAudioEngine(): Promise<boolean> {
  const engine = getAudioEngine();
  return await engine.init();
}

export default AudioEngine;
