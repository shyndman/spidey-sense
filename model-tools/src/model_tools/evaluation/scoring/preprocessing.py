"""Fixed scoring image transform and probability kernel."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from model_tools.metadata import InputMetadata

PROBABILITY_COUNT = 1_000


def softmax_logits(logits: NDArray[np.float32]) -> tuple[float, ...]:
    if logits.shape != (PROBABILITY_COUNT,) or not bool(np.all(np.isfinite(logits))):
        raise ValueError("invalid logits")
    shifted: NDArray[np.float64] = logits.astype(np.float64) - float(np.max(logits))
    exponentials: NDArray[np.float64] = np.exp(shifted)
    total = float(np.sum(exponentials, dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid softmax sum")
    probabilities: NDArray[np.float64] = exponentials / total
    if not bool(np.all(np.isfinite(probabilities))):
        raise ValueError("invalid softmax probabilities")
    return tuple(float(value) for value in probabilities.flat)


def preprocess_image(path: Path, metadata: InputMetadata) -> NDArray[np.float32]:
    if not math.isfinite(metadata.pixel_scale) or metadata.pixel_scale <= 0:
        raise ValueError("invalid pixel scale")
    _, channels, target_height, target_width = metadata.shape
    if channels != 3:
        raise ValueError("evaluation input must have three channels")
    with Image.open(path) as source:
        _ = source.load()
        image = source.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("invalid image dimensions")
    scale = min(target_width / image.width, target_height / image.height)
    width = min(target_width, max(1, math.floor(image.width * scale + 0.5)))
    height = min(target_height, max(1, math.floor(image.height * scale + 0.5)))
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    contained = np.zeros((target_height, target_width, channels), dtype=np.float32)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    contained[top : top + height, left : left + width] = pixels
    contained *= np.float32(metadata.pixel_scale)
    mean = np.asarray(metadata.mean, dtype=np.float32)
    deviation = np.asarray(metadata.standard_deviation, dtype=np.float32)
    if not bool(np.all(np.isfinite(mean))) or not bool(
        np.all(np.isfinite(deviation) & (deviation > 0))
    ):
        raise ValueError("invalid normalization")
    normalized = (contained - mean) / deviation
    tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    if tensor.shape != (1, channels, target_height, target_width) or not bool(
        np.all(np.isfinite(tensor))
    ):
        raise ValueError("invalid preprocessed tensor")
    return np.ascontiguousarray(tensor, dtype=np.float32)
