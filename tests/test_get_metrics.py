import numpy as np
import pytest

from evaluation.get_metrics import (
    compute_aggregated_top3_accuracy,
    compute_distances,
    indices_of_smallest,
    majority_vote,
)


def test_compute_distances_returns_euclidean_matrix():
    embeddings = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])

    distances = compute_distances(embeddings)

    assert distances.shape == (3, 3)
    assert distances[0, 0] == 0.0
    assert distances[0, 1] == 5.0
    assert distances[1, 2] == 5.0


def test_majority_vote_returns_most_common_rank():
    assert majority_vote([2, 1, 2, 3, 2]) == 2


def test_aggregated_top3_accuracy_detects_ground_truth():
    ranks_per_frame = [[4, 3, 1], [2, 1, 5], [1, 6, 7]]

    assert compute_aggregated_top3_accuracy(ranks_per_frame, ground_truth=1) == 1
    assert compute_aggregated_top3_accuracy(ranks_per_frame, ground_truth=9) == 0


def test_aggregated_top3_accuracy_requires_votes():
    with pytest.raises(AssertionError, match="No votes collected"):
        compute_aggregated_top3_accuracy([[], [1, 2]], ground_truth=1)


def test_indices_of_smallest_excludes_banned_index():
    distances = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    ranks = indices_of_smallest(distances, banned_idx=0)

    assert ranks == [1, 2, 3]
