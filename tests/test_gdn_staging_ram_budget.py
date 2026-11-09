# SPDX-License-Identifier: Apache-2.0
"""gdn_staging_ram_budget: RAM-first staging of prefill boundary snapshots.

In the embedded GDN layout with the SSD tier on, prefill boundary snapshots
stay in RAM, in one pool shared by all in-flight requests, up to the budget
(a percentage of the hot cache, or a size) instead of being staged to SSD;
beyond it they spill to the existing SSD staging store. "0" = always stage
to SSD (the previous behavior).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.scheduler import Scheduler, _tree_nbytes
from omlx.settings import GlobalSettings


def _sched(*, split, store=True, budget=0, fraction=0.0, hot_max=0, prompt=10 * 2048 + 300):
    stub = SimpleNamespace(
        config=SimpleNamespace(gdn_staging_ram_budget_bytes=budget, gdn_staging_ram_budget_fraction=fraction),
        block_aware_cache=SimpleNamespace(_gdn_snapshot_interval=1),
        paged_ssd_cache_manager=SimpleNamespace(
            _expected_layer_cache_types=["KVCache", "ArraysCache"],
            _effective_hot_cache_max_bytes=lambda: hot_max,
        ),
        requests={"r": SimpleNamespace(num_prompt_tokens=prompt)},
        _gdn_split_active=lambda: split,
        _boundary_snapshot_store=object() if store else None,
    )
    stub._staging_ram_budget_bytes = lambda: Scheduler._staging_ram_budget_bytes(stub)
    return stub


def test_ram_first_active_only_for_embedded_with_store_and_budget():
    active = Scheduler._ram_first_staging_active
    assert active(_sched(split=False, budget=1 << 30))
    assert not active(_sched(split=True, budget=1 << 30))
    assert not active(_sched(split=False, budget=0))
    assert not active(_sched(split=False, budget=1 << 30, store=False))


def test_budget_as_fraction_of_hot_cache():
    budget = Scheduler._staging_ram_budget_bytes
    assert budget(_sched(split=False, fraction=0.3, hot_max=10_000)) == 3_000
    assert budget(_sched(split=False, fraction=0.3, hot_max=0)) == 0  # no hot cache: off
    assert budget(_sched(split=False, budget=5_000)) == 5_000  # explicit size
    assert budget(_sched(split=False, budget=5_000, fraction=0.5, hot_max=4_000)) == 2_000  # fraction wins
    assert not Scheduler._ram_first_staging_active(_sched(split=False, fraction=0.3, hot_max=0))
    assert Scheduler._ram_first_staging_active(_sched(split=False, fraction=0.3, hot_max=10_000))


def test_tree_nbytes_counts_nested_arrays():
    value = ("__prefill_extracted__", [{"state": (mx.zeros((10,), dtype=mx.float32), [mx.zeros((5,), dtype=mx.float16)]),
                                        "meta_state": (3,)}])
    assert _tree_nbytes(value) == 10 * 4 + 5 * 2


class _FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, request_id, token_count, snapshot_cache, extract_fn, *, block_size=None):
        self.saved.append(token_count)
        return True


def _capture_stub(budget, nbytes_per_snapshot):
    elems = nbytes_per_snapshot // 4
    stub = _sched(split=False, budget=budget)
    store = _FakeStore()
    stub._boundary_snapshot_store = store
    stub._model_has_unreconstructible_cache = lambda: False
    stub._boundary_snapshot_diagnostics = MagicMock()
    stub._cache_list_needs_boundary_snapshot = lambda cache: True
    stub._boundary_cache_snapshots = {}
    stub._prefill_snapshot_value = lambda cache: (
        "__prefill_extracted__", [{"state": (mx.zeros((elems,), dtype=mx.float32),), "class_name": "ArraysCache"}]
    )
    stub._stream = mx.default_stream(mx.default_device())
    stub._enable_mtp_boundary_alignment = lambda: None
    stub._extract_snapshot_cache_states = lambda c: ([], None)
    stub._sidecar_boundaries_skipped = 0
    stub._staging_ram_kept = 0
    stub._staging_ram_kept_bytes = 0
    stub._staging_spilled = 0
    stub._boundary_grid_interval = lambda: 1
    stub._sidecar_boundary_needed = lambda rid, tc, bs: True
    stub._ram_first_staging_active = lambda: Scheduler._ram_first_staging_active(stub)
    stub.config.paged_cache_block_size = 2048
    return stub, store


def test_snapshots_stay_in_ram_under_budget_then_spill():
    stub, store = _capture_stub(budget=3 * 4000, nbytes_per_snapshot=4000)
    for t in (2048, 4096, 6144, 8192):
        Scheduler._on_prefill_boundary_snapshot(stub, "r", [object()], t)
    snaps = stub._boundary_cache_snapshots["r"]
    assert [t for t, v in snaps.items() if v is not None] == [2048, 4096, 6144]  # in RAM
    assert snaps[8192] is None and store.saved == [8192]  # spilled to SSD staging
    assert stub._staging_ram_kept == 3 and stub._staging_spilled == 1


def test_budget_is_shared_across_in_flight_requests():
    stub, store = _capture_stub(budget=3 * 4000, nbytes_per_snapshot=4000)
    for t in (2048, 4096):
        Scheduler._on_prefill_boundary_snapshot(stub, "r", [object()], t)
    stub.requests["q"] = SimpleNamespace(num_prompt_tokens=10 * 2048 + 300)
    for t in (2048, 4096):
        Scheduler._on_prefill_boundary_snapshot(stub, "q", [object()], t)
    snaps = stub._boundary_cache_snapshots
    assert snaps["q"][2048] is not None  # third snapshot overall still fits
    assert snaps["q"][4096] is None and store.saved == [4096]  # fourth spills
    assert stub._staging_ram_kept == 3 and stub._staging_spilled == 1


def test_budget_zero_keeps_historical_ssd_staging():
    stub, store = _capture_stub(budget=0, nbytes_per_snapshot=4000)
    for t in (2048, 4096):
        Scheduler._on_prefill_boundary_snapshot(stub, "r", [object()], t)
    assert store.saved == [2048, 4096]
    assert all(v is None for v in stub._boundary_cache_snapshots["r"].values())


def test_settings_roundtrip_and_validation(tmp_path):
    s = GlobalSettings(base_path=tmp_path)
    assert s.cache.gdn_staging_ram_budget == "30%"
    default_config = s.to_scheduler_config()
    assert default_config.gdn_staging_ram_budget_fraction == pytest.approx(0.3)
    assert default_config.gdn_staging_ram_budget_bytes == 0
    s.cache.gdn_staging_ram_budget = "1GB"
    s.save()
    loaded = GlobalSettings.load(base_path=tmp_path)
    assert loaded.cache.gdn_staging_ram_budget == "1GB"
    assert loaded.to_scheduler_config().gdn_staging_ram_budget_bytes == 1024**3
    assert loaded.to_scheduler_config().gdn_staging_ram_budget_fraction == 0.0
    for bad in ("lots", "150%", "x%", "-1GB"):
        loaded.cache.gdn_staging_ram_budget = bad
        assert any("gdn_staging_ram_budget" in e for e in loaded.validate()), bad
    loaded.cache.gdn_staging_ram_budget = "0"
    assert loaded.to_scheduler_config().gdn_staging_ram_budget_bytes == 0
