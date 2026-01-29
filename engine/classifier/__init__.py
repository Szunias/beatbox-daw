"""Beatbox sound classifier module."""

from .model import (
    BeatboxCNN,
    BeatboxCNNLite,
    DRUM_CLASSES,
    CLASS_NAMES,
    NUM_CLASSES,
    create_model,
)
from .inference import (
    BeatboxClassifier,
    RuleBasedClassifier,
    InferenceConfig,
)

__all__ = [
    'BeatboxCNN',
    'BeatboxCNNLite',
    'DRUM_CLASSES',
    'CLASS_NAMES',
    'NUM_CLASSES',
    'create_model',
    'BeatboxClassifier',
    'RuleBasedClassifier',
    'InferenceConfig',
]
