"""Shared fixtures for the test suite.

Keeps the per-test boilerplate (`MagicMock` image tensors, the four-line
setup for `images.shape`/`images.__getitem__`) out of every test that needs
a fake ComfyUI image batch.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_image_batch(count, size=32):
    """Return a MagicMock batch of `count` 32x32x3 float tensors.

    Mimics the parts of a torch tensor the node actually uses: `.shape`,
    `[i]` indexing, `.cpu().numpy()`. The numpy payload is real so any test
    that lets PIL actually decode a frame exercises a real encoder path.
    """
    tensor = MagicMock()
    tensor.cpu.return_value = tensor
    tensor.numpy.return_value = np.random.rand(size, size, 3).astype(np.float32)

    images = MagicMock()
    images.shape = [count]
    images.__getitem__ = lambda _self, i: tensor
    return images


@pytest.fixture
def image_batch():
    """Single-image batch — the most common shape in the suite."""
    return _make_image_batch(1)


@pytest.fixture
def image_batch_of_two():
    """Two-image batch for batch-containment tests."""
    return _make_image_batch(2)


@pytest.fixture
def image_batch_of_three():
    """Three-image batch for batch-containment tests."""
    return _make_image_batch(3)
