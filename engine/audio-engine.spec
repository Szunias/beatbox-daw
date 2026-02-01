# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for BeatBox DAW Audio Engine.

Bundles the Python audio processing engine into a standalone executable
with proper hooks for torch, numpy, scipy, librosa, and sounddevice.

Usage:
    cd engine
    pyinstaller audio-engine.spec

Output:
    dist/audio-engine.exe (Windows)
    dist/audio-engine (Linux/macOS)
"""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules
import sys
import os

# Determine if we're on Windows
is_windows = sys.platform.startswith('win')

# ============================================================================
# DATA FILES
# ============================================================================
# Collect data files required by torch and torchaudio for model loading
datas = []
datas += collect_data_files('torch')
datas += collect_data_files('torchaudio')
datas += collect_data_files('librosa')

# Include our classifier weights directory (if weights exist)
# This ensures trained model weights are bundled with the executable
datas += [('classifier/weights', 'classifier/weights')]

# ============================================================================
# METADATA
# ============================================================================
# Copy package metadata needed for version detection at runtime
datas += copy_metadata('torch')
datas += copy_metadata('torchaudio')
datas += copy_metadata('numpy')
datas += copy_metadata('scipy')
datas += copy_metadata('librosa')
datas += copy_metadata('sounddevice')
datas += copy_metadata('websockets')

# ============================================================================
# HIDDEN IMPORTS
# ============================================================================
# These modules are imported dynamically and not detected by PyInstaller
hiddenimports = [
    # NumPy internals
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy.fft',
    'numpy.linalg',
    'numpy.random',

    # SciPy special functions (often missed)
    'scipy.special._ufuncs',
    'scipy.special._cdflib',
    'scipy._lib.messagestream',
    'scipy.fft._pocketfft',
    'scipy.signal.windows',

    # PyTorch internals
    'torch._C',
    'torch.utils.data',
    'torch.nn.functional',

    # Torchaudio (often dynamically loaded)
    'torchaudio',
    'torchaudio.transforms',

    # Sounddevice backend
    'sounddevice',
    '_sounddevice_data',

    # MIDI libraries
    'mido',
    'mido.backends',
    'mido.backends.rtmidi',
    'rtmidi',

    # WebSocket
    'websockets',
    'websockets.server',
    'websockets.exceptions',

    # Standard library often missed
    'asyncio',
    'threading',
    'queue',
    'dataclasses',
    'json',
    'typing',
    'enum',
    'pathlib',
    'collections',
]

# Collect ALL scipy submodules (scipy has many dynamically loaded parts)
hiddenimports += collect_submodules('scipy')

# Collect librosa submodules (for audio feature extraction)
hiddenimports += collect_submodules('librosa')

# Collect sounddevice submodules
hiddenimports += collect_submodules('sounddevice')

# Collect torch submodules (selective to avoid bloat)
hiddenimports += [
    'torch.nn',
    'torch.nn.modules',
    'torch.nn.modules.conv',
    'torch.nn.modules.batchnorm',
    'torch.nn.modules.pooling',
    'torch.nn.modules.linear',
    'torch.nn.modules.dropout',
    'torch.nn.modules.activation',
    'torch.optim',
    'torch.autograd',
    'torch.cuda',
]

# ============================================================================
# BINARIES
# ============================================================================
# Additional binary files (native libraries)
binaries = []

# Sounddevice requires portaudio library
# PyInstaller usually handles this, but we include it explicitly if needed
# On Windows, the DLL is typically bundled with sounddevice package

# ============================================================================
# EXCLUDES
# ============================================================================
# Modules to exclude (reduce size, avoid conflicts)
excludes = [
    'tkinter',           # GUI toolkit (not needed)
    'matplotlib',        # Plotting (not needed for runtime)
    'IPython',           # Interactive Python (not needed)
    'notebook',          # Jupyter (not needed)
    'pytest',            # Testing framework
    'pytest_asyncio',    # Async testing
    'sphinx',            # Documentation
    'docutils',          # Documentation
    '_pytest',           # Testing internals
    'test',              # Test modules
    'tests',             # Test directories
]

# ============================================================================
# ANALYSIS
# ============================================================================
a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# ============================================================================
# PYZ (Python bytecode archive)
# ============================================================================
pyz = PYZ(a.pure)

# ============================================================================
# EXE (Final executable)
# ============================================================================
# Build as single file executable (--onefile mode)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='audio-engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # Enable UPX compression if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,            # Keep console for logging/debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,               # Add icon path here if desired: icon='path/to/icon.ico'
)

# ============================================================================
# NOTES
# ============================================================================
#
# Build command:
#   cd engine
#   pyinstaller audio-engine.spec
#
# The output will be:
#   dist/audio-engine.exe (Windows)
#   dist/audio-engine (Linux/macOS)
#
# To reduce executable size:
#   1. Use CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu
#   2. This reduces size from ~500MB to ~150MB
#
# After building:
#   1. Test standalone: ./dist/audio-engine.exe (should start WebSocket server on port 8765)
#   2. Copy to Tauri binaries:
#      - Windows: src-tauri/binaries/audio-engine-x86_64-pc-windows-msvc.exe
#      - Linux: src-tauri/binaries/audio-engine-x86_64-unknown-linux-gnu
#      - macOS: src-tauri/binaries/audio-engine-x86_64-apple-darwin
#
