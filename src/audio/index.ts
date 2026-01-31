/**
 * Audio Module
 * Web Audio API based audio playback for the DAW
 */

export { AudioEngine, getAudioEngine, initAudioEngine } from './AudioEngine';
export type { default as AudioEngineType } from './AudioEngine';

export {
  EffectsProcessor,
  getEffectsProcessor,
  createEffectsProcessor,
} from './EffectsProcessor';

export type {
  EffectType,
  BaseEffectParameters,
  EQ3BandParameters,
  CompressorParameters,
  DelayParameters,
  ReverbParameters,
  EffectParameters,
} from './EffectsProcessor';
