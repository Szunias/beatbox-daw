"""Audio effects processing for real-time DSP effects."""

import numpy as np
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import signal


@dataclass
class EffectConfig:
    """Base configuration for audio effects."""
    sample_rate: int = 44100
    buffer_size: int = 512
    enabled: bool = True
    mix: float = 1.0  # Dry/wet mix (0-1)


# === Base Effect Class ===

class BaseEffect(ABC):
    """Abstract base class for audio effects."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.enabled = True
        self.mix = 1.0  # Dry/wet mix (0-1)

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio buffer through the effect.

        Args:
            audio: Audio samples (mono or stereo, float32)

        Returns:
            Processed audio samples
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset effect state (clear delay lines, filters, etc.)."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current effect parameters."""
        pass

    @abstractmethod
    def set_parameter(self, name: str, value: float) -> bool:
        """Set an effect parameter by name."""
        pass

    def apply_mix(self, dry: np.ndarray, wet: np.ndarray) -> np.ndarray:
        """Apply dry/wet mix to processed audio."""
        if self.mix >= 1.0:
            return wet
        elif self.mix <= 0.0:
            return dry
        return dry * (1.0 - self.mix) + wet * self.mix


# === 3-Band Equalizer ===

@dataclass
class EQ3BandConfig:
    """Configuration for 3-band equalizer."""
    sample_rate: int = 44100
    low_freq: float = 200.0      # Low band cutoff frequency
    high_freq: float = 3000.0    # High band cutoff frequency
    low_gain: float = 0.0        # Low band gain in dB (-12 to +12)
    mid_gain: float = 0.0        # Mid band gain in dB (-12 to +12)
    high_gain: float = 0.0       # High band gain in dB (-12 to +12)


class EQ3Band(BaseEffect):
    """3-band parametric equalizer with low, mid, and high frequency bands."""

    def __init__(self, config: Optional[EQ3BandConfig] = None):
        self.config = config or EQ3BandConfig()
        super().__init__(self.config.sample_rate)

        # Initialize filter coefficients
        self._design_filters()

        # Filter states for biquad processing
        self._low_state = np.zeros(2)
        self._high_state = np.zeros(2)

    def _design_filters(self) -> None:
        """Design the crossover filters."""
        nyquist = self.sample_rate / 2.0

        # Ensure frequencies are valid
        low_freq = min(self.config.low_freq, nyquist * 0.9)
        high_freq = min(self.config.high_freq, nyquist * 0.9)

        # Design lowpass for low band separation
        try:
            self._low_b, self._low_a = signal.butter(
                2, low_freq / nyquist, btype='low'
            )
        except Exception:
            # Fallback to safe values
            self._low_b = np.array([1.0])
            self._low_a = np.array([1.0])

        # Design highpass for high band separation
        try:
            self._high_b, self._high_a = signal.butter(
                2, high_freq / nyquist, btype='high'
            )
        except Exception:
            # Fallback to safe values
            self._high_b = np.array([1.0])
            self._high_a = np.array([1.0])

    def _db_to_linear(self, db: float) -> float:
        """Convert decibels to linear gain."""
        return 10.0 ** (db / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through the 3-band EQ."""
        if not self.enabled:
            return audio

        # Handle stereo by processing each channel
        if audio.ndim == 2:
            left = self.process(audio[:, 0])
            right = self.process(audio[:, 1])
            return np.column_stack([left, right])

        # Separate bands using crossover filters
        low_band, self._low_state = signal.lfilter(
            self._low_b, self._low_a, audio, zi=self._low_state
        )
        high_band, self._high_state = signal.lfilter(
            self._high_b, self._high_a, audio, zi=self._high_state
        )

        # Mid band is what's left after removing low and high
        mid_band = audio - low_band - high_band

        # Apply gains
        low_gain = self._db_to_linear(self.config.low_gain)
        mid_gain = self._db_to_linear(self.config.mid_gain)
        high_gain = self._db_to_linear(self.config.high_gain)

        # Sum bands with gains
        wet = low_band * low_gain + mid_band * mid_gain + high_band * high_gain

        return self.apply_mix(audio, wet)

    def reset(self) -> None:
        """Reset filter states."""
        self._low_state = np.zeros(2)
        self._high_state = np.zeros(2)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current EQ parameters."""
        return {
            'low_freq': self.config.low_freq,
            'high_freq': self.config.high_freq,
            'low_gain': self.config.low_gain,
            'mid_gain': self.config.mid_gain,
            'high_gain': self.config.high_gain,
            'enabled': self.enabled,
            'mix': self.mix,
        }

    def set_parameter(self, name: str, value: float) -> bool:
        """Set an EQ parameter by name."""
        if name == 'low_freq':
            self.config.low_freq = max(20.0, min(value, 500.0))
            self._design_filters()
            return True
        elif name == 'high_freq':
            self.config.high_freq = max(1000.0, min(value, 16000.0))
            self._design_filters()
            return True
        elif name == 'low_gain':
            self.config.low_gain = max(-12.0, min(value, 12.0))
            return True
        elif name == 'mid_gain':
            self.config.mid_gain = max(-12.0, min(value, 12.0))
            return True
        elif name == 'high_gain':
            self.config.high_gain = max(-12.0, min(value, 12.0))
            return True
        elif name == 'enabled':
            self.enabled = bool(value)
            return True
        elif name == 'mix':
            self.mix = max(0.0, min(value, 1.0))
            return True
        return False


# === Compressor ===

@dataclass
class CompressorConfig:
    """Configuration for dynamics compressor."""
    sample_rate: int = 44100
    threshold: float = -20.0     # Threshold in dB
    ratio: float = 4.0           # Compression ratio (e.g., 4:1)
    attack_ms: float = 10.0      # Attack time in milliseconds
    release_ms: float = 100.0    # Release time in milliseconds
    makeup_gain: float = 0.0     # Makeup gain in dB
    knee: float = 6.0            # Soft knee width in dB


class Compressor(BaseEffect):
    """Dynamics compressor with soft knee and makeup gain."""

    def __init__(self, config: Optional[CompressorConfig] = None):
        self.config = config or CompressorConfig()
        super().__init__(self.config.sample_rate)

        # Calculate time constants
        self._update_time_constants()

        # Envelope follower state
        self._envelope = 0.0

    def _update_time_constants(self) -> None:
        """Calculate attack and release coefficients from time constants."""
        # Time constants for exponential envelope
        if self.config.attack_ms > 0:
            self._attack_coef = np.exp(-1.0 / (self.config.attack_ms * self.sample_rate / 1000.0))
        else:
            self._attack_coef = 0.0

        if self.config.release_ms > 0:
            self._release_coef = np.exp(-1.0 / (self.config.release_ms * self.sample_rate / 1000.0))
        else:
            self._release_coef = 0.0

    def _db_to_linear(self, db: float) -> float:
        """Convert decibels to linear gain."""
        return 10.0 ** (db / 20.0)

    def _linear_to_db(self, linear: float) -> float:
        """Convert linear gain to decibels."""
        if linear <= 0:
            return -96.0  # Floor at -96 dB
        return 20.0 * np.log10(linear)

    def _compute_gain_reduction(self, input_db: float) -> float:
        """Compute gain reduction for a given input level in dB."""
        threshold = self.config.threshold
        ratio = self.config.ratio
        knee = self.config.knee

        # Below threshold
        if input_db < threshold - knee / 2:
            return 0.0

        # Above threshold + knee
        if input_db > threshold + knee / 2:
            return (threshold - input_db) * (1.0 - 1.0 / ratio)

        # In knee region - smooth transition
        knee_factor = ((input_db - threshold + knee / 2) ** 2) / (2 * knee)
        return knee_factor * (1.0 / ratio - 1.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through the compressor."""
        if not self.enabled:
            return audio

        # Handle stereo by processing both channels with the same envelope
        if audio.ndim == 2:
            # Use the max of both channels for envelope
            mono = np.maximum(np.abs(audio[:, 0]), np.abs(audio[:, 1]))
            # Process envelope detection on mono
            gain = self._process_mono(mono, compute_only=True)
            # Apply same gain to both channels
            wet = audio * gain.reshape(-1, 1)
            return self.apply_mix(audio, wet)

        # Mono processing
        gain = self._process_mono(np.abs(audio), compute_only=True)
        wet = audio * gain

        return self.apply_mix(audio, wet)

    def _process_mono(self, audio_abs: np.ndarray, compute_only: bool = False) -> np.ndarray:
        """Process mono audio and return gain reduction curve."""
        output_gain = np.ones(len(audio_abs))

        for i in range(len(audio_abs)):
            # Get input level
            input_level = audio_abs[i]

            # Update envelope follower
            if input_level > self._envelope:
                self._envelope = self._attack_coef * self._envelope + (1 - self._attack_coef) * input_level
            else:
                self._envelope = self._release_coef * self._envelope + (1 - self._release_coef) * input_level

            # Convert to dB
            envelope_db = self._linear_to_db(self._envelope)

            # Compute gain reduction
            gain_reduction_db = self._compute_gain_reduction(envelope_db)

            # Apply makeup gain
            total_gain_db = gain_reduction_db + self.config.makeup_gain

            # Convert back to linear
            output_gain[i] = self._db_to_linear(total_gain_db)

        return output_gain

    def reset(self) -> None:
        """Reset compressor state."""
        self._envelope = 0.0

    def get_parameters(self) -> Dict[str, Any]:
        """Get current compressor parameters."""
        return {
            'threshold': self.config.threshold,
            'ratio': self.config.ratio,
            'attack_ms': self.config.attack_ms,
            'release_ms': self.config.release_ms,
            'makeup_gain': self.config.makeup_gain,
            'knee': self.config.knee,
            'enabled': self.enabled,
            'mix': self.mix,
        }

    def set_parameter(self, name: str, value: float) -> bool:
        """Set a compressor parameter by name."""
        if name == 'threshold':
            self.config.threshold = max(-60.0, min(value, 0.0))
            return True
        elif name == 'ratio':
            self.config.ratio = max(1.0, min(value, 20.0))
            return True
        elif name == 'attack_ms':
            self.config.attack_ms = max(0.1, min(value, 500.0))
            self._update_time_constants()
            return True
        elif name == 'release_ms':
            self.config.release_ms = max(1.0, min(value, 5000.0))
            self._update_time_constants()
            return True
        elif name == 'makeup_gain':
            self.config.makeup_gain = max(0.0, min(value, 24.0))
            return True
        elif name == 'knee':
            self.config.knee = max(0.0, min(value, 24.0))
            return True
        elif name == 'enabled':
            self.enabled = bool(value)
            return True
        elif name == 'mix':
            self.mix = max(0.0, min(value, 1.0))
            return True
        return False


# === Delay ===

@dataclass
class DelayConfig:
    """Configuration for delay effect."""
    sample_rate: int = 44100
    delay_ms: float = 250.0      # Delay time in milliseconds
    feedback: float = 0.3        # Feedback amount (0-0.95)
    mix: float = 0.3             # Wet mix amount (0-1)


class Delay(BaseEffect):
    """Simple delay/echo effect with feedback."""

    def __init__(self, config: Optional[DelayConfig] = None):
        self.config = config or DelayConfig()
        super().__init__(self.config.sample_rate)
        self.mix = self.config.mix

        # Calculate buffer size for max 2 seconds of delay
        self._max_delay_samples = int(2.0 * self.sample_rate)
        self._delay_buffer = np.zeros(self._max_delay_samples)
        self._write_pos = 0

        self._update_delay_samples()

    def _update_delay_samples(self) -> None:
        """Update delay time in samples."""
        self._delay_samples = int(self.config.delay_ms * self.sample_rate / 1000.0)
        self._delay_samples = min(self._delay_samples, self._max_delay_samples - 1)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through the delay."""
        if not self.enabled:
            return audio

        # Handle stereo by processing each channel
        if audio.ndim == 2:
            left = self.process(audio[:, 0])
            right = self.process(audio[:, 1])
            return np.column_stack([left, right])

        output = np.zeros_like(audio)
        feedback = min(self.config.feedback, 0.95)  # Prevent runaway feedback

        for i in range(len(audio)):
            # Read from delay buffer
            read_pos = (self._write_pos - self._delay_samples) % self._max_delay_samples
            delayed = self._delay_buffer[read_pos]

            # Write to delay buffer (input + feedback)
            self._delay_buffer[self._write_pos] = audio[i] + delayed * feedback

            # Output is dry + wet
            output[i] = audio[i] + delayed * self.mix

            # Advance write position
            self._write_pos = (self._write_pos + 1) % self._max_delay_samples

        return output

    def reset(self) -> None:
        """Reset delay buffer."""
        self._delay_buffer.fill(0.0)
        self._write_pos = 0

    def get_parameters(self) -> Dict[str, Any]:
        """Get current delay parameters."""
        return {
            'delay_ms': self.config.delay_ms,
            'feedback': self.config.feedback,
            'enabled': self.enabled,
            'mix': self.mix,
        }

    def set_parameter(self, name: str, value: float) -> bool:
        """Set a delay parameter by name."""
        if name == 'delay_ms':
            self.config.delay_ms = max(1.0, min(value, 2000.0))
            self._update_delay_samples()
            return True
        elif name == 'feedback':
            self.config.feedback = max(0.0, min(value, 0.95))
            return True
        elif name == 'enabled':
            self.enabled = bool(value)
            return True
        elif name == 'mix':
            self.mix = max(0.0, min(value, 1.0))
            return True
        return False


# === Reverb (Simple Algorithmic) ===

@dataclass
class ReverbConfig:
    """Configuration for reverb effect."""
    sample_rate: int = 44100
    room_size: float = 0.5       # Room size (0-1)
    damping: float = 0.5         # High frequency damping (0-1)
    mix: float = 0.3             # Wet mix amount (0-1)


class Reverb(BaseEffect):
    """Simple algorithmic reverb using comb and allpass filters."""

    def __init__(self, config: Optional[ReverbConfig] = None):
        self.config = config or ReverbConfig()
        super().__init__(self.config.sample_rate)
        self.mix = self.config.mix

        # Comb filter delay times (in samples, scaled for 44100Hz)
        scale = self.sample_rate / 44100.0
        self._comb_delays = [
            int(1557 * scale), int(1617 * scale),
            int(1491 * scale), int(1422 * scale),
            int(1277 * scale), int(1356 * scale),
            int(1188 * scale), int(1116 * scale),
        ]

        # Allpass filter delay times
        self._allpass_delays = [
            int(225 * scale), int(556 * scale),
            int(441 * scale), int(341 * scale),
        ]

        # Initialize comb filter buffers
        self._comb_buffers = [np.zeros(delay) for delay in self._comb_delays]
        self._comb_pos = [0] * len(self._comb_delays)
        self._comb_filter_store = [0.0] * len(self._comb_delays)

        # Initialize allpass filter buffers
        self._allpass_buffers = [np.zeros(delay) for delay in self._allpass_delays]
        self._allpass_pos = [0] * len(self._allpass_delays)

    def _process_comb(self, input_sample: float, index: int) -> float:
        """Process a single sample through a comb filter."""
        delay = self._comb_delays[index]
        buffer = self._comb_buffers[index]
        pos = self._comb_pos[index]

        # Read from buffer
        output = buffer[pos]

        # Damping filter
        filter_store = self._comb_filter_store[index]
        filtered = output * (1 - self.config.damping) + filter_store * self.config.damping
        self._comb_filter_store[index] = filtered

        # Feedback with room size
        feedback = self.config.room_size * 0.98  # Scale room size
        buffer[pos] = input_sample + filtered * feedback

        # Advance position
        self._comb_pos[index] = (pos + 1) % delay

        return output

    def _process_allpass(self, input_sample: float, index: int) -> float:
        """Process a single sample through an allpass filter."""
        delay = self._allpass_delays[index]
        buffer = self._allpass_buffers[index]
        pos = self._allpass_pos[index]

        # Read from buffer
        buffered = buffer[pos]

        # Allpass coefficient
        coef = 0.5

        output = -input_sample + buffered
        buffer[pos] = input_sample + buffered * coef

        # Advance position
        self._allpass_pos[index] = (pos + 1) % delay

        return output

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through the reverb."""
        if not self.enabled:
            return audio

        # Handle stereo by converting to mono for reverb, then mix back
        if audio.ndim == 2:
            mono = (audio[:, 0] + audio[:, 1]) * 0.5
            wet = self._process_mono(mono)
            # Mix wet signal back to stereo
            wet_stereo = np.column_stack([wet, wet])
            return self.apply_mix(audio, wet_stereo)

        wet = self._process_mono(audio)
        return self.apply_mix(audio, wet)

    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        """Process mono audio through reverb."""
        output = np.zeros_like(audio)

        for i in range(len(audio)):
            sample = audio[i]

            # Sum of comb filters (parallel)
            comb_sum = 0.0
            for j in range(len(self._comb_delays)):
                comb_sum += self._process_comb(sample, j)
            comb_sum /= len(self._comb_delays)

            # Series allpass filters
            allpass_out = comb_sum
            for j in range(len(self._allpass_delays)):
                allpass_out = self._process_allpass(allpass_out, j)

            output[i] = allpass_out

        return output

    def reset(self) -> None:
        """Reset reverb buffers."""
        for buffer in self._comb_buffers:
            buffer.fill(0.0)
        for buffer in self._allpass_buffers:
            buffer.fill(0.0)
        self._comb_pos = [0] * len(self._comb_delays)
        self._allpass_pos = [0] * len(self._allpass_delays)
        self._comb_filter_store = [0.0] * len(self._comb_delays)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current reverb parameters."""
        return {
            'room_size': self.config.room_size,
            'damping': self.config.damping,
            'enabled': self.enabled,
            'mix': self.mix,
        }

    def set_parameter(self, name: str, value: float) -> bool:
        """Set a reverb parameter by name."""
        if name == 'room_size':
            self.config.room_size = max(0.0, min(value, 1.0))
            return True
        elif name == 'damping':
            self.config.damping = max(0.0, min(value, 1.0))
            return True
        elif name == 'enabled':
            self.enabled = bool(value)
            return True
        elif name == 'mix':
            self.mix = max(0.0, min(value, 1.0))
            return True
        return False


# === Effects Processor (Effect Chain Manager) ===

@dataclass
class EffectsProcessorConfig:
    """Configuration for the effects processor."""
    sample_rate: int = 44100
    buffer_size: int = 512


class EffectsProcessor:
    """
    Manages a chain of audio effects for a track or bus.

    Effects are processed in order, with each effect's output
    feeding into the next effect's input.
    """

    # Registry of available effect types
    EFFECT_TYPES: Dict[str, type] = {
        'eq3band': EQ3Band,
        'compressor': Compressor,
        'delay': Delay,
        'reverb': Reverb,
    }

    def __init__(self, config: Optional[EffectsProcessorConfig] = None):
        self.config = config or EffectsProcessorConfig()
        self.sample_rate = self.config.sample_rate

        # Effect chain (list of effects in processing order)
        self._effects: List[BaseEffect] = []
        self._effect_ids: List[str] = []

        # Bypass all effects
        self._bypassed = False

    def add_effect(self, effect_type: str, effect_id: Optional[str] = None,
                   position: Optional[int] = None) -> Optional[str]:
        """
        Add an effect to the chain.

        Args:
            effect_type: Type of effect ('eq3band', 'compressor', 'delay', 'reverb')
            effect_id: Optional ID for the effect (auto-generated if not provided)
            position: Optional position in chain (appended if not provided)

        Returns:
            Effect ID if successful, None if effect type not found
        """
        if effect_type not in self.EFFECT_TYPES:
            return None

        # Create effect instance
        effect_class = self.EFFECT_TYPES[effect_type]

        # Create appropriate config based on effect type
        if effect_type == 'eq3band':
            effect = effect_class(EQ3BandConfig(sample_rate=self.sample_rate))
        elif effect_type == 'compressor':
            effect = effect_class(CompressorConfig(sample_rate=self.sample_rate))
        elif effect_type == 'delay':
            effect = effect_class(DelayConfig(sample_rate=self.sample_rate))
        elif effect_type == 'reverb':
            effect = effect_class(ReverbConfig(sample_rate=self.sample_rate))
        else:
            effect = effect_class()

        # Generate ID if not provided
        if effect_id is None:
            effect_id = f"{effect_type}_{len(self._effects)}_{id(effect)}"

        # Add to chain
        if position is None or position >= len(self._effects):
            self._effects.append(effect)
            self._effect_ids.append(effect_id)
        else:
            self._effects.insert(position, effect)
            self._effect_ids.insert(position, effect_id)

        return effect_id

    def remove_effect(self, effect_id: str) -> bool:
        """
        Remove an effect from the chain.

        Args:
            effect_id: ID of effect to remove

        Returns:
            True if effect was removed, False if not found
        """
        try:
            index = self._effect_ids.index(effect_id)
            self._effects.pop(index)
            self._effect_ids.pop(index)
            return True
        except ValueError:
            return False

    def move_effect(self, effect_id: str, new_position: int) -> bool:
        """
        Move an effect to a new position in the chain.

        Args:
            effect_id: ID of effect to move
            new_position: New position in chain

        Returns:
            True if effect was moved, False if not found
        """
        try:
            old_index = self._effect_ids.index(effect_id)
            effect = self._effects.pop(old_index)
            eid = self._effect_ids.pop(old_index)

            # Adjust position if needed
            new_position = max(0, min(new_position, len(self._effects)))

            self._effects.insert(new_position, effect)
            self._effect_ids.insert(new_position, eid)
            return True
        except ValueError:
            return False

    def get_effect(self, effect_id: str) -> Optional[BaseEffect]:
        """Get an effect by ID."""
        try:
            index = self._effect_ids.index(effect_id)
            return self._effects[index]
        except ValueError:
            return None

    def set_effect_parameter(self, effect_id: str, param_name: str, value: float) -> bool:
        """
        Set a parameter on an effect.

        Args:
            effect_id: ID of effect
            param_name: Name of parameter
            value: New value

        Returns:
            True if parameter was set, False if effect not found or param invalid
        """
        effect = self.get_effect(effect_id)
        if effect is None:
            return False
        return effect.set_parameter(param_name, value)

    def get_effect_parameters(self, effect_id: str) -> Optional[Dict[str, Any]]:
        """Get all parameters of an effect."""
        effect = self.get_effect(effect_id)
        if effect is None:
            return None
        return effect.get_parameters()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through the effect chain.

        Args:
            audio: Audio samples (mono or stereo, float32)

        Returns:
            Processed audio samples
        """
        if self._bypassed or len(self._effects) == 0:
            return audio

        result = audio.copy()

        for effect in self._effects:
            if effect.enabled:
                result = effect.process(result)

        return result

    def reset(self) -> None:
        """Reset all effects in the chain."""
        for effect in self._effects:
            effect.reset()

    def bypass(self, bypassed: bool = True) -> None:
        """Set bypass state for entire chain."""
        self._bypassed = bypassed

    def is_bypassed(self) -> bool:
        """Check if chain is bypassed."""
        return self._bypassed

    def get_chain_info(self) -> List[Dict[str, Any]]:
        """Get information about all effects in the chain."""
        info = []
        for effect_id, effect in zip(self._effect_ids, self._effects):
            effect_type = None
            for name, cls in self.EFFECT_TYPES.items():
                if isinstance(effect, cls):
                    effect_type = name
                    break

            info.append({
                'id': effect_id,
                'type': effect_type,
                'enabled': effect.enabled,
                'parameters': effect.get_parameters(),
            })
        return info

    def clear(self) -> None:
        """Remove all effects from the chain."""
        self._effects.clear()
        self._effect_ids.clear()

    @property
    def effect_count(self) -> int:
        """Get number of effects in chain."""
        return len(self._effects)


if __name__ == "__main__":
    # Test effects processor
    import time

    # Create test signal
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create a simple sine wave with some harmonics (like a pluck)
    test_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.25 * np.sin(2 * np.pi * 880 * t) +
        0.125 * np.sin(2 * np.pi * 1320 * t)
    )
    # Apply envelope
    envelope = np.exp(-3 * t)
    test_signal = test_signal * envelope

    # Test EffectsProcessor
    processor = EffectsProcessor(EffectsProcessorConfig(sample_rate=sr))

    # Add effects
    eq_id = processor.add_effect('eq3band')
    comp_id = processor.add_effect('compressor')

    # Configure EQ
    processor.set_effect_parameter(eq_id, 'low_gain', 3.0)
    processor.set_effect_parameter(eq_id, 'high_gain', -2.0)

    # Configure compressor
    processor.set_effect_parameter(comp_id, 'threshold', -15.0)
    processor.set_effect_parameter(comp_id, 'ratio', 4.0)

    # Process in chunks
    buffer_size = 512
    processed = []

    start_time = time.perf_counter()
    for i in range(0, len(test_signal), buffer_size):
        chunk = test_signal[i:i + buffer_size]
        if len(chunk) < buffer_size:
            chunk = np.pad(chunk, (0, buffer_size - len(chunk)))
        processed.append(processor.process(chunk))
    elapsed = time.perf_counter() - start_time

    # Combine processed chunks
    processed_signal = np.concatenate(processed)[:len(test_signal)]

    # Print results
    print(f"Processed {len(test_signal)} samples in {elapsed*1000:.2f}ms")
    print(f"Real-time factor: {duration / elapsed:.1f}x")
    print(f"Effect chain: {[info['type'] for info in processor.get_chain_info()]}")
    print(f"Input RMS: {np.sqrt(np.mean(test_signal**2)):.4f}")
    print(f"Output RMS: {np.sqrt(np.mean(processed_signal**2)):.4f}")

    # Test individual effects
    print("\n--- Testing EQ3Band ---")
    eq = EQ3Band()
    eq_params = eq.get_parameters()
    print(f"EQ Parameters: {eq_params}")

    print("\n--- Testing Compressor ---")
    comp = Compressor()
    comp_params = comp.get_parameters()
    print(f"Compressor Parameters: {comp_params}")

    print("\nEffects imported successfully")
