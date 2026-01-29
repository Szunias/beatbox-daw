# BeatBox DAW

A digital audio workstation with real-time beatbox-to-MIDI conversion powered by machine learning.

## Overview

BeatBox DAW captures your vocal beatbox performances and converts them to MIDI in real-time. The application uses onset detection and sound classification to identify drum sounds (kick, snare, hi-hat, clap, tom) and maps them to corresponding MIDI notes for use in any DAW or virtual instrument.

## Live Demo

Try the web demo at: `https://[username].github.io/beatbox-daw/`

The web version runs in demo mode with simulated patterns since the Python audio backend requires local hardware. Download the desktop application for full beatbox detection functionality.

## Features

### Beatbox-to-MIDI Conversion
- Real-time audio capture with low-latency processing
- Onset detection using spectral flux analysis
- ML-based sound classification for five drum types
- MIDI output to virtual ports or external DAWs
- Waveform visualization with event feedback

### DAW Functionality
- Multi-track timeline with drag-and-drop arrangement
- Transport controls with loop regions
- Piano roll editor with snap-to-grid and velocity editing
- Per-track mixer with volume, pan, mute, and solo
- Project save and load capabilities

### Supported Drum Classes

| Sound | MIDI Note | Description |
|-------|-----------|-------------|
| Kick | 36 | Bass drum |
| Snare | 38 | Snare drum |
| Hi-hat | 42 | Closed hi-hat |
| Clap | 39 | Hand clap |
| Tom | 45 | Low tom |

## Quick Start

### Web Demo

1. Visit the GitHub Pages URL
2. Enter an access code when prompted
3. Click "BeatBox" to open the panel
4. Select a demo pattern and click "Play Demo"

### Desktop Application

1. Install prerequisites (Node.js 18+, Python 3.10+, Rust)
2. Clone the repository
3. Install dependencies:
   ```
   npm install
   cd engine && pip install -r requirements.txt
   ```
4. Start the Python audio engine:
   ```
   python engine/main.py
   ```
5. Run the application:
   ```
   npm run tauri:dev
   ```

## Requirements

### Desktop Application
- Node.js 18 or later
- Python 3.10 or later
- Rust toolchain (for Tauri)
- Audio input device (microphone)
- Optional: Virtual MIDI port (loopMIDI on Windows, IAC Driver on macOS)

### Web Demo
- Modern browser with JavaScript enabled
- No additional requirements

## Architecture

```
Frontend (React/TypeScript)
    |
    | WebSocket
    v
Audio Engine (Python)
    |
    +-- Audio Capture (sounddevice)
    +-- Onset Detection (librosa)
    +-- Sound Classification (PyTorch)
    +-- MIDI Output (mido/rtmidi)
```

The frontend provides the DAW interface and communicates with the Python backend via WebSocket. The backend handles audio capture, signal processing, and MIDI output.

## Building

### Web Build
```
npm run build
```

Output is generated in the `dist` directory.

### Desktop Build
```
npm run tauri:build
```

Platform-specific installers are generated in `src-tauri/target/release/bundle`.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Enter | Stop |
| R | Toggle Record |
| M | Toggle Mixer |
| B | Toggle BeatBox Panel |
| Delete | Delete selected notes |
| Escape | Close Piano Roll |

## Project Structure

```
beatbox-daw/
├── src/                    # React frontend
│   ├── components/         # UI components
│   ├── hooks/              # Custom React hooks
│   ├── stores/             # Zustand state stores
│   └── types/              # TypeScript definitions
├── engine/                 # Python audio backend
│   ├── main.py             # WebSocket server
│   ├── audio_capture.py    # Audio input
│   ├── onset_detector.py   # Beat detection
│   └── classifier/         # ML classification
└── src-tauri/              # Tauri Rust backend
```

## Configuration

### Audio Settings

Edit `engine/main.py` to adjust:

```python
config = EngineConfig(
    sample_rate=44100,
    buffer_size=512,
    confidence_threshold=0.5
)
```

Lower buffer sizes reduce latency but increase CPU usage.

### ML Classifier

To use the trained model instead of rule-based detection:

```python
config = EngineConfig(use_ml_classifier=True)
```

## Troubleshooting

### No audio input detected
- Verify microphone permissions in system settings
- Check the correct input device is selected in Settings
- Run `python engine/audio_capture.py` to list available devices

### High latency
- Reduce buffer size in configuration
- Close other audio applications
- Use a dedicated audio interface

### MIDI not working
- Install a virtual MIDI driver if needed
- Verify MIDI routing in your target DAW
- Test with `python engine/midi_output.py`

## License

MIT
