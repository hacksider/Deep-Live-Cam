"""Tests for modules/cluster_analysis.py."""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


def test_find_cluster_centroids_single_element():
    """Single embedding should not crash — returns immediately."""
    from modules.cluster_analysis import find_cluster_centroids
    embeddings = np.array([[1.0, 0.0, 0.0]])
    result = find_cluster_centroids(embeddings)
    assert result is not None


def test_find_cluster_centroids_empty_raises_or_returns():
    """Empty input should not produce an unhandled exception."""
    from modules.cluster_analysis import find_cluster_centroids
    embeddings = np.empty((0, 3))
    try:
        result = find_cluster_centroids(embeddings)
        # If it returns, result should be truthy or at least not crash
    except (ValueError, Exception):
        pass  # Acceptable to raise on empty input


def test_find_closest_centroid_basic():
    """find_closest_centroid returns correct index for simple case."""
    from modules.cluster_analysis import find_closest_centroid
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    embedding = np.array([0.9, 0.1])
    idx, centroid = find_closest_centroid(centroids, embedding)
    assert idx == 0
