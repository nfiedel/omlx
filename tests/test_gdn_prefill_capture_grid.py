# SPDX-License-Identifier: Apache-2.0
"""Prefill captures follow the gdn_snapshot_interval grid.

With interval N > 1, a store keeps real recurrent state only on the N-block
grid and at its last block, so prefill captures only grid boundaries and the
prompt's final full-block boundary (tails have their own capture path). In
the sidecar layout this always applies; in the embedded layout only when the
prefix cache would thin the layout (ArraysCache-family state).
"""

from __future__ import annotations

from types import SimpleNamespace

from omlx.scheduler import Scheduler


def _sched(*, split, interval=1, applies=True, prompt=10 * 2048 + 300):
    prefix = SimpleNamespace(
        _gdn_snapshot_interval=interval,
        _snapshot_thinning_applies=lambda *a, **k: applies,
    )
    stub = SimpleNamespace(
        block_aware_cache=prefix,
        paged_ssd_cache_manager=SimpleNamespace(_expected_layer_cache_types=["KVCache", "ArraysCache"]),
        requests={"r": SimpleNamespace(num_prompt_tokens=prompt)},
        _gdn_split_active=lambda: split,
    )
    stub._boundary_grid_interval = lambda: Scheduler._boundary_grid_interval(stub)
    return stub


def _staged(stub):
    return [t for t in range(2048, 10 * 2048 + 1, 2048) if Scheduler._sidecar_boundary_needed(stub, "r", t, 2048)]


FULL = list(range(2048, 10 * 2048 + 1, 2048))
GRID_4 = [4 * 2048, 8 * 2048, 10 * 2048]  # grid + final full-block boundary


def test_embedded_capture_follows_snapshot_grid():
    assert _staged(_sched(split=False, interval=4)) == GRID_4


def test_embedded_capture_unthinned_when_not_applicable():
    assert _staged(_sched(split=False, interval=1)) == FULL
    assert _staged(_sched(split=False, interval=4, applies=False)) == FULL


def test_split_capture_uses_the_same_interval():
    assert _staged(_sched(split=True, interval=1)) == FULL
    assert _staged(_sched(split=True, interval=4)) == GRID_4
    # The embedded-layout applicability check does not gate the sidecar layout.
    assert _staged(_sched(split=True, interval=4, applies=False)) == GRID_4
