/**
 * EffectsProcessor
 * Web Audio API based effects processing for the DAW
 * Manages effect chains with EQ, Compressor, Delay, and Reverb
 */

// Effect types matching the backend
export type EffectType = 'eq3band' | 'compressor' | 'delay' | 'reverb';

// === Effect Parameter Interfaces ===

// Base interface with common properties for all effects
export interface BaseEffectParameters {
  enabled: boolean;
  mix: number;          // Dry/wet mix (0-1)
}

export interface EQ3BandParameters extends BaseEffectParameters {
  lowFreq: number;      // Low band cutoff frequency (20-500 Hz)
  highFreq: number;     // High band cutoff frequency (1000-16000 Hz)
  lowGain: number;      // Low band gain in dB (-12 to +12)
  midGain: number;      // Mid band gain in dB (-12 to +12)
  highGain: number;     // High band gain in dB (-12 to +12)
}

export interface CompressorParameters extends BaseEffectParameters {
  threshold: number;    // Threshold in dB (-60 to 0)
  ratio: number;        // Compression ratio (1 to 20)
  attack: number;       // Attack time in seconds (0.001 to 1)
  release: number;      // Release time in seconds (0.01 to 1)
  knee: number;         // Knee width in dB (0 to 40)
}

export interface DelayParameters extends BaseEffectParameters {
  delayTime: number;    // Delay time in seconds (0.001 to 2)
  feedback: number;     // Feedback amount (0 to 0.95)
}

export interface ReverbParameters extends BaseEffectParameters {
  decay: number;        // Decay time (0.1 to 10 seconds)
  preDelay: number;     // Pre-delay time (0 to 0.1 seconds)
}

export type EffectParameters =
  | EQ3BandParameters
  | CompressorParameters
  | DelayParameters
  | ReverbParameters;

// === Default Parameters ===

const DEFAULT_EQ3BAND_PARAMS: EQ3BandParameters = {
  lowFreq: 200,
  highFreq: 3000,
  lowGain: 0,
  midGain: 0,
  highGain: 0,
  enabled: true,
  mix: 1,
};

const DEFAULT_COMPRESSOR_PARAMS: CompressorParameters = {
  threshold: -20,
  ratio: 4,
  attack: 0.01,
  release: 0.1,
  knee: 6,
  enabled: true,
  mix: 1,
};

const DEFAULT_DELAY_PARAMS: DelayParameters = {
  delayTime: 0.25,
  feedback: 0.3,
  enabled: true,
  mix: 0.3,
};

const DEFAULT_REVERB_PARAMS: ReverbParameters = {
  decay: 2,
  preDelay: 0.02,
  enabled: true,
  mix: 0.3,
};

// === Effect Instance Interface ===

interface EffectInstance {
  id: string;
  type: EffectType;
  params: EffectParameters;
  nodes: AudioNode[];
  inputNode: AudioNode;
  outputNode: AudioNode;
  dryGainNode: GainNode;
  wetGainNode: GainNode;
  enabled: boolean;
}

// === EffectsProcessor Class ===

export class EffectsProcessor {
  private audioContext: AudioContext | null = null;
  private effects: Map<string, EffectInstance> = new Map();
  private effectOrder: string[] = [];
  private inputNode: GainNode | null = null;
  private outputNode: GainNode | null = null;
  private bypassed: boolean = false;
  private bypassNode: GainNode | null = null;

  /**
   * Initialize the effects processor with an audio context
   */
  init(audioContext: AudioContext): void {
    this.audioContext = audioContext;

    // Create input/output gain nodes
    this.inputNode = audioContext.createGain();
    this.outputNode = audioContext.createGain();
    this.bypassNode = audioContext.createGain();

    // Connect input to output directly (bypass path)
    this.inputNode.connect(this.bypassNode);
    this.bypassNode.connect(this.outputNode);
    this.bypassNode.gain.value = 0; // Effects are active by default

    // When there are no effects, connect input directly to output
    this.inputNode.connect(this.outputNode);
  }

  /**
   * Get the input node for connecting audio sources
   */
  getInputNode(): GainNode | null {
    return this.inputNode;
  }

  /**
   * Get the output node for connecting to destination
   */
  getOutputNode(): GainNode | null {
    return this.outputNode;
  }

  /**
   * Add an effect to the chain
   */
  addEffect(
    effectType: EffectType,
    effectId?: string,
    position?: number
  ): string | null {
    if (!this.audioContext) {
      return null;
    }

    // Generate ID if not provided
    const id = effectId || `${effectType}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Create the effect based on type
    let effect: EffectInstance | null = null;

    switch (effectType) {
      case 'eq3band':
        effect = this.createEQ3Band(id);
        break;
      case 'compressor':
        effect = this.createCompressor(id);
        break;
      case 'delay':
        effect = this.createDelay(id);
        break;
      case 'reverb':
        effect = this.createReverb(id);
        break;
    }

    if (!effect) {
      return null;
    }

    // Store effect
    this.effects.set(id, effect);

    // Add to order at specified position
    if (position !== undefined && position >= 0 && position < this.effectOrder.length) {
      this.effectOrder.splice(position, 0, id);
    } else {
      this.effectOrder.push(id);
    }

    // Rebuild the effect chain
    this.rebuildChain();

    return id;
  }

  /**
   * Remove an effect from the chain
   */
  removeEffect(effectId: string): boolean {
    const effect = this.effects.get(effectId);
    if (!effect) {
      return false;
    }

    // Disconnect all nodes
    effect.nodes.forEach(node => {
      node.disconnect();
    });
    effect.dryGainNode.disconnect();
    effect.wetGainNode.disconnect();

    // Remove from maps
    this.effects.delete(effectId);
    this.effectOrder = this.effectOrder.filter(id => id !== effectId);

    // Rebuild chain
    this.rebuildChain();

    return true;
  }

  /**
   * Move an effect to a new position in the chain
   */
  moveEffect(effectId: string, newPosition: number): boolean {
    const currentIndex = this.effectOrder.indexOf(effectId);
    if (currentIndex === -1) {
      return false;
    }

    // Remove from current position
    this.effectOrder.splice(currentIndex, 1);

    // Insert at new position
    const insertPos = Math.max(0, Math.min(newPosition, this.effectOrder.length));
    this.effectOrder.splice(insertPos, 0, effectId);

    // Rebuild chain
    this.rebuildChain();

    return true;
  }

  /**
   * Set a parameter on an effect
   */
  setEffectParameter(effectId: string, paramName: string, value: number): boolean {
    const effect = this.effects.get(effectId);
    if (!effect) {
      return false;
    }

    // Update the parameter based on effect type
    const params = effect.params;
    if (!(paramName in params)) {
      return false;
    }

    // Use type-safe property assignment
    switch (effect.type) {
      case 'eq3band':
        (params as EQ3BandParameters)[paramName as keyof EQ3BandParameters] = value as never;
        break;
      case 'compressor':
        (params as CompressorParameters)[paramName as keyof CompressorParameters] = value as never;
        break;
      case 'delay':
        (params as DelayParameters)[paramName as keyof DelayParameters] = value as never;
        break;
      case 'reverb':
        (params as ReverbParameters)[paramName as keyof ReverbParameters] = value as never;
        break;
    }

    // Apply the parameter to the actual audio nodes
    this.applyEffectParameters(effect);

    return true;
  }

  /**
   * Get all parameters of an effect
   */
  getEffectParameters(effectId: string): EffectParameters | null {
    const effect = this.effects.get(effectId);
    if (!effect) {
      return null;
    }
    return { ...effect.params };
  }

  /**
   * Enable or disable an effect
   */
  setEffectEnabled(effectId: string, enabled: boolean): boolean {
    const effect = this.effects.get(effectId);
    if (!effect) {
      return false;
    }

    effect.enabled = enabled;
    effect.params.enabled = enabled;

    // Adjust wet/dry mix to bypass effect
    if (enabled) {
      effect.wetGainNode.gain.value = effect.params.mix;
      effect.dryGainNode.gain.value = 1 - effect.params.mix;
    } else {
      effect.wetGainNode.gain.value = 0;
      effect.dryGainNode.gain.value = 1;
    }

    return true;
  }

  /**
   * Bypass the entire effect chain
   */
  setBypass(bypassed: boolean): void {
    this.bypassed = bypassed;

    if (!this.inputNode || !this.bypassNode) return;

    if (bypassed) {
      // Disconnect effects and use bypass path
      this.bypassNode.gain.value = 1;
      // Mute the effect chain output
      this.effectOrder.forEach(id => {
        const effect = this.effects.get(id);
        if (effect) {
          effect.outputNode.disconnect();
        }
      });
    } else {
      // Reconnect effects
      this.bypassNode.gain.value = 0;
      this.rebuildChain();
    }
  }

  /**
   * Check if the chain is bypassed
   */
  isBypassed(): boolean {
    return this.bypassed;
  }

  /**
   * Get information about all effects in the chain
   */
  getChainInfo(): Array<{
    id: string;
    type: EffectType;
    enabled: boolean;
    parameters: EffectParameters;
  }> {
    return this.effectOrder.map(id => {
      const effect = this.effects.get(id)!;
      return {
        id: effect.id,
        type: effect.type,
        enabled: effect.enabled,
        parameters: { ...effect.params },
      };
    });
  }

  /**
   * Clear all effects from the chain
   */
  clear(): void {
    // Disconnect all effects
    this.effects.forEach(effect => {
      effect.nodes.forEach(node => node.disconnect());
      effect.dryGainNode.disconnect();
      effect.wetGainNode.disconnect();
    });

    this.effects.clear();
    this.effectOrder = [];

    // Reconnect input to output
    if (this.inputNode && this.outputNode) {
      this.inputNode.disconnect();
      this.inputNode.connect(this.outputNode);
      this.inputNode.connect(this.bypassNode!);
    }
  }

  /**
   * Reset all effects (clear delay lines, etc.)
   */
  reset(): void {
    // For Web Audio API, effects are stateless or manage their own state
    // We can recreate delay buffers by briefly disconnecting
    this.effects.forEach(effect => {
      if (effect.type === 'delay') {
        const delayNode = effect.nodes.find(n => n instanceof DelayNode) as DelayNode;
        if (delayNode) {
          // Reset by setting delay to 0 briefly
          const currentDelay = (effect.params as DelayParameters).delayTime;
          delayNode.delayTime.value = 0;
          delayNode.delayTime.value = currentDelay;
        }
      }
    });
  }

  /**
   * Get the number of effects in the chain
   */
  get effectCount(): number {
    return this.effects.size;
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    this.clear();

    if (this.inputNode) {
      this.inputNode.disconnect();
      this.inputNode = null;
    }
    if (this.outputNode) {
      this.outputNode.disconnect();
      this.outputNode = null;
    }
    if (this.bypassNode) {
      this.bypassNode.disconnect();
      this.bypassNode = null;
    }

    this.audioContext = null;
  }

  // === Private Methods ===

  /**
   * Create a 3-band EQ effect
   */
  private createEQ3Band(id: string): EffectInstance | null {
    if (!this.audioContext) return null;

    const params = { ...DEFAULT_EQ3BAND_PARAMS };

    // Create low shelf filter
    const lowShelf = this.audioContext.createBiquadFilter();
    lowShelf.type = 'lowshelf';
    lowShelf.frequency.value = params.lowFreq;
    lowShelf.gain.value = params.lowGain;

    // Create mid peaking filter
    const midPeak = this.audioContext.createBiquadFilter();
    midPeak.type = 'peaking';
    midPeak.frequency.value = Math.sqrt(params.lowFreq * params.highFreq);
    midPeak.Q.value = 1;
    midPeak.gain.value = params.midGain;

    // Create high shelf filter
    const highShelf = this.audioContext.createBiquadFilter();
    highShelf.type = 'highshelf';
    highShelf.frequency.value = params.highFreq;
    highShelf.gain.value = params.highGain;

    // Create dry/wet mix nodes
    const dryGain = this.audioContext.createGain();
    const wetGain = this.audioContext.createGain();
    const inputSplit = this.audioContext.createGain();
    const outputMix = this.audioContext.createGain();

    // Connect: input -> split -> [dry, wet chain] -> output
    dryGain.gain.value = 1 - params.mix;
    wetGain.gain.value = params.mix;

    // Connect wet path: lowShelf -> midPeak -> highShelf
    lowShelf.connect(midPeak);
    midPeak.connect(highShelf);
    highShelf.connect(wetGain);
    wetGain.connect(outputMix);

    // Dry path
    dryGain.connect(outputMix);

    return {
      id,
      type: 'eq3band',
      params,
      nodes: [lowShelf, midPeak, highShelf, inputSplit, outputMix],
      inputNode: inputSplit,
      outputNode: outputMix,
      dryGainNode: dryGain,
      wetGainNode: wetGain,
      enabled: true,
    };
  }

  /**
   * Create a compressor effect
   */
  private createCompressor(id: string): EffectInstance | null {
    if (!this.audioContext) return null;

    const params = { ...DEFAULT_COMPRESSOR_PARAMS };

    // Create Web Audio compressor
    const compressor = this.audioContext.createDynamicsCompressor();
    compressor.threshold.value = params.threshold;
    compressor.ratio.value = params.ratio;
    compressor.attack.value = params.attack;
    compressor.release.value = params.release;
    compressor.knee.value = params.knee;

    // Create dry/wet mix nodes
    const dryGain = this.audioContext.createGain();
    const wetGain = this.audioContext.createGain();
    const inputSplit = this.audioContext.createGain();
    const outputMix = this.audioContext.createGain();

    dryGain.gain.value = 1 - params.mix;
    wetGain.gain.value = params.mix;

    // Wet path
    compressor.connect(wetGain);
    wetGain.connect(outputMix);

    // Dry path
    dryGain.connect(outputMix);

    return {
      id,
      type: 'compressor',
      params,
      nodes: [compressor, inputSplit, outputMix],
      inputNode: inputSplit,
      outputNode: outputMix,
      dryGainNode: dryGain,
      wetGainNode: wetGain,
      enabled: true,
    };
  }

  /**
   * Create a delay effect
   */
  private createDelay(id: string): EffectInstance | null {
    if (!this.audioContext) return null;

    const params = { ...DEFAULT_DELAY_PARAMS };

    // Create delay node
    const delayNode = this.audioContext.createDelay(2); // Max 2 seconds
    delayNode.delayTime.value = params.delayTime;

    // Create feedback gain
    const feedbackGain = this.audioContext.createGain();
    feedbackGain.gain.value = params.feedback;

    // Create dry/wet mix nodes
    const dryGain = this.audioContext.createGain();
    const wetGain = this.audioContext.createGain();
    const inputSplit = this.audioContext.createGain();
    const outputMix = this.audioContext.createGain();

    dryGain.gain.value = 1 - params.mix;
    wetGain.gain.value = params.mix;

    // Connect delay with feedback
    delayNode.connect(feedbackGain);
    feedbackGain.connect(delayNode);
    delayNode.connect(wetGain);
    wetGain.connect(outputMix);

    // Dry path
    dryGain.connect(outputMix);

    return {
      id,
      type: 'delay',
      params,
      nodes: [delayNode, feedbackGain, inputSplit, outputMix],
      inputNode: inputSplit,
      outputNode: outputMix,
      dryGainNode: dryGain,
      wetGainNode: wetGain,
      enabled: true,
    };
  }

  /**
   * Create a reverb effect using convolution
   */
  private createReverb(id: string): EffectInstance | null {
    if (!this.audioContext) return null;

    const params = { ...DEFAULT_REVERB_PARAMS };

    // Create pre-delay
    const preDelay = this.audioContext.createDelay(0.1);
    preDelay.delayTime.value = params.preDelay;

    // Create convolver for reverb
    const convolver = this.audioContext.createConvolver();

    // Generate impulse response
    this.generateImpulseResponse(convolver, params.decay);

    // Create dry/wet mix nodes
    const dryGain = this.audioContext.createGain();
    const wetGain = this.audioContext.createGain();
    const inputSplit = this.audioContext.createGain();
    const outputMix = this.audioContext.createGain();

    dryGain.gain.value = 1 - params.mix;
    wetGain.gain.value = params.mix;

    // Wet path: preDelay -> convolver
    preDelay.connect(convolver);
    convolver.connect(wetGain);
    wetGain.connect(outputMix);

    // Dry path
    dryGain.connect(outputMix);

    return {
      id,
      type: 'reverb',
      params,
      nodes: [preDelay, convolver, inputSplit, outputMix],
      inputNode: inputSplit,
      outputNode: outputMix,
      dryGainNode: dryGain,
      wetGainNode: wetGain,
      enabled: true,
    };
  }

  /**
   * Generate an impulse response for reverb
   */
  private generateImpulseResponse(convolver: ConvolverNode, decay: number): void {
    if (!this.audioContext) return;

    const sampleRate = this.audioContext.sampleRate;
    const length = Math.floor(sampleRate * decay);
    const impulse = this.audioContext.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const channelData = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        // Exponentially decaying noise
        const decayFactor = Math.exp(-3 * i / length);
        channelData[i] = (Math.random() * 2 - 1) * decayFactor;
      }
    }

    convolver.buffer = impulse;
  }

  /**
   * Apply effect parameters to audio nodes
   */
  private applyEffectParameters(effect: EffectInstance): void {
    if (!this.audioContext) return;

    switch (effect.type) {
      case 'eq3band': {
        const params = effect.params as EQ3BandParameters;
        const [lowShelf, midPeak, highShelf] = effect.nodes as BiquadFilterNode[];

        lowShelf.frequency.value = params.lowFreq;
        lowShelf.gain.value = params.lowGain;

        midPeak.frequency.value = Math.sqrt(params.lowFreq * params.highFreq);
        midPeak.gain.value = params.midGain;

        highShelf.frequency.value = params.highFreq;
        highShelf.gain.value = params.highGain;

        effect.wetGainNode.gain.value = params.enabled ? params.mix : 0;
        effect.dryGainNode.gain.value = params.enabled ? 1 - params.mix : 1;
        break;
      }
      case 'compressor': {
        const params = effect.params as CompressorParameters;
        const compressor = effect.nodes[0] as DynamicsCompressorNode;

        compressor.threshold.value = params.threshold;
        compressor.ratio.value = params.ratio;
        compressor.attack.value = params.attack;
        compressor.release.value = params.release;
        compressor.knee.value = params.knee;

        effect.wetGainNode.gain.value = params.enabled ? params.mix : 0;
        effect.dryGainNode.gain.value = params.enabled ? 1 - params.mix : 1;
        break;
      }
      case 'delay': {
        const params = effect.params as DelayParameters;
        const delayNode = effect.nodes[0] as DelayNode;
        const feedbackGain = effect.nodes[1] as GainNode;

        delayNode.delayTime.value = params.delayTime;
        feedbackGain.gain.value = Math.min(params.feedback, 0.95);

        effect.wetGainNode.gain.value = params.enabled ? params.mix : 0;
        effect.dryGainNode.gain.value = params.enabled ? 1 - params.mix : 1;
        break;
      }
      case 'reverb': {
        const params = effect.params as ReverbParameters;
        const preDelay = effect.nodes[0] as DelayNode;
        const convolver = effect.nodes[1] as ConvolverNode;

        preDelay.delayTime.value = params.preDelay;

        // Regenerate impulse response if decay changed significantly
        if (convolver.buffer && Math.abs(convolver.buffer.duration - params.decay) > 0.5) {
          this.generateImpulseResponse(convolver, params.decay);
        }

        effect.wetGainNode.gain.value = params.enabled ? params.mix : 0;
        effect.dryGainNode.gain.value = params.enabled ? 1 - params.mix : 1;
        break;
      }
    }
  }

  /**
   * Rebuild the effect chain connections
   */
  private rebuildChain(): void {
    if (!this.inputNode || !this.outputNode || !this.audioContext) return;

    // Disconnect input from everything except bypass
    this.inputNode.disconnect();
    this.inputNode.connect(this.bypassNode!);

    if (this.effectOrder.length === 0) {
      // No effects, connect input directly to output
      this.inputNode.connect(this.outputNode);
      return;
    }

    // Connect effects in order
    let previousOutput: AudioNode = this.inputNode;

    for (const effectId of this.effectOrder) {
      const effect = this.effects.get(effectId);
      if (!effect) continue;

      // Disconnect effect nodes first
      effect.inputNode.disconnect();
      effect.dryGainNode.disconnect();
      effect.wetGainNode.disconnect();
      effect.outputNode.disconnect();

      // Connect input split
      previousOutput.connect(effect.inputNode);

      // Connect dry path
      effect.inputNode.connect(effect.dryGainNode);
      effect.dryGainNode.connect(effect.outputNode);

      // Connect wet path based on effect type
      switch (effect.type) {
        case 'eq3band': {
          const [lowShelf] = effect.nodes as BiquadFilterNode[];
          effect.inputNode.connect(lowShelf);
          break;
        }
        case 'compressor': {
          const compressor = effect.nodes[0] as DynamicsCompressorNode;
          effect.inputNode.connect(compressor);
          break;
        }
        case 'delay': {
          const delayNode = effect.nodes[0] as DelayNode;
          effect.inputNode.connect(delayNode);
          break;
        }
        case 'reverb': {
          const preDelay = effect.nodes[0] as DelayNode;
          effect.inputNode.connect(preDelay);
          break;
        }
      }

      previousOutput = effect.outputNode;
    }

    // Connect last effect to output
    previousOutput.connect(this.outputNode);
  }
}

// === Singleton Instance ===

let effectsProcessorInstance: EffectsProcessor | null = null;

/**
 * Get the singleton EffectsProcessor instance
 */
export function getEffectsProcessor(): EffectsProcessor {
  if (!effectsProcessorInstance) {
    effectsProcessorInstance = new EffectsProcessor();
  }
  return effectsProcessorInstance;
}

/**
 * Create a new EffectsProcessor instance (for per-track use)
 */
export function createEffectsProcessor(): EffectsProcessor {
  return new EffectsProcessor();
}

export default EffectsProcessor;
