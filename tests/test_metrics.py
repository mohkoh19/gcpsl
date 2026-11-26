import math
from itertools import combinations

import pytest

from gcpsl.metrics import ParticipationTracker, Target, gini


def test_participation_factorization_and_gaps():
    tracker = ParticipationTracker(4)
    for active, updated in [({0, 1}, {0, 1, 2, 3}), ({1}, {0, 1}), ({0, 2, 3}, {0, 1, 2, 3})]:
        tracker.add_phase(active, updated)
    summary = tracker.summary()
    assert summary["direct_participation"] == pytest.approx(6 / 12)
    assert summary["model_update_participation"] == pytest.approx(10 / 12)
    assert summary["within_round_participation"] == pytest.approx(6 / 10)
    assert summary["coverage"] == 1.0
    assert summary["mean_gap"] == pytest.approx((0 + 1) / 2)


def test_gini():
    assert gini([1, 1, 1]) == pytest.approx(0)
    assert gini([0, 0, 3]) == pytest.approx(2 / 3)


def test_target_records_the_first_crossing():
    target = Target(0.5)
    for epoch, accuracy in enumerate([0.4, 0.6, 0.3, 0.7], 1):
        target.update(accuracy=accuracy, epoch=epoch, seconds=10.0 * epoch)
    assert (target.epoch, target.seconds) == (2, 20.0)


def test_exact_participation_probability():
    """Closed-form participation probability of the sampler against exhaustive enumeration."""
    counts = [2, 3, 4]
    pool = [k for k, c in enumerate(counts) for _ in range(c)]
    for size in range(1, len(pool) + 1):
        draws = list(combinations(range(len(pool)), size))
        empirical = sum(len({pool[i] for i in draw}) for draw in draws) / len(draws)
        exact = sum(
            1 - math.comb(len(pool) - c, size) / math.comb(len(pool), size) for c in counts
        )
        assert empirical == pytest.approx(exact)
