# SPDX-License-Identifier: Apache-2.0
"""Tests for the dev-only cache experiment controls (OMLX_CACHE_EXPERIMENTS=1)."""

import asyncio
import concurrent.futures
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — triggers set_admin_getters
from omlx.admin import cache_experiment as ce


def _scheduler(**overrides):
    prefix = MagicMock()
    prefix.clear = MagicMock(return_value=7)
    ssd = SimpleNamespace(
        clear_hot_cache=MagicMock(return_value=3),
        _evict_profile={"tail:never_reused": {"count": 1}},
        _hot_cache_max_bytes=8 << 30,
        _hot_cache_total_bytes=0,
    )
    sched = SimpleNamespace(
        waiting=[],
        prefilling=[],
        running={},
        _inflight_store_futures={},
        block_aware_cache=prefix,
        paged_ssd_cache_manager=ssd,
        config=SimpleNamespace(hot_cache_max_size=8 << 30, paged_cache_block_size=2048),
    )
    for k, v in overrides.items():
        setattr(sched, k, v)
    return sched


def _pool():
    budget = SimpleNamespace(max_bytes=8 << 30, total_bytes=0)
    return SimpleNamespace(
        _scheduler_config=SimpleNamespace(hot_cache_max_size=8 << 30, hot_cache_budget=budget)
    )


class TestHelpers:
    def test_experiments_flag(self, monkeypatch):
        monkeypatch.delenv(ce.ENV_FLAG, raising=False)
        assert not ce.experiments_enabled()
        for value in ("1", "true", "on"):
            monkeypatch.setenv(ce.ENV_FLAG, value)
            assert ce.experiments_enabled()
        monkeypatch.setenv(ce.ENV_FLAG, "0")
        assert not ce.experiments_enabled()

    def test_busy_detection(self):
        assert ce.scheduler_busy(_scheduler()) is None
        assert "running" in ce.scheduler_busy(_scheduler(running={"r": 1}))
        assert "waiting" in ce.scheduler_busy(_scheduler(waiting=["r"]))
        pending = concurrent.futures.Future()
        assert "stores" in ce.scheduler_busy(
            _scheduler(_inflight_store_futures={"r": pending})
        )
        done = concurrent.futures.Future()
        done.set_result(None)
        assert ce.scheduler_busy(_scheduler(_inflight_store_futures={"r": done})) is None

    def test_wait_until_idle_times_out(self):
        busy = ce.wait_until_idle([("m", _scheduler(running={"r": 1}))], timeout_s=0.0)
        assert busy == {"m": "running requests present"}

    def test_reset_clears_prefix_state_hot_tier_and_profile(self):
        sched = _scheduler()
        out = ce.reset_prefix_state(sched)
        assert out == {"prefix_entries": 7, "hot_entries": 3}
        sched.block_aware_cache.clear.assert_called_once()
        sched.paged_ssd_cache_manager.clear_hot_cache.assert_called_once()
        assert sched.paged_ssd_cache_manager._evict_profile == {}

    def test_apply_hot_cache_max_bytes_updates_every_layer(self):
        pool, sched = _pool(), _scheduler()
        ce.apply_hot_cache_max_bytes(pool, [("m", sched)], 2 << 30)
        assert pool._scheduler_config.hot_cache_max_size == 2 << 30
        assert pool._scheduler_config.hot_cache_budget.max_bytes == 2 << 30
        assert sched.config.hot_cache_max_size == 2 << 30
        assert sched.paged_ssd_cache_manager._hot_cache_max_bytes == 2 << 30


def _post(body, sched=None, pool=None):
    sched = sched or _scheduler()
    pool = pool or _pool()
    with patch.object(admin_routes, "_iter_loaded_schedulers", lambda: iter([("m", sched)])), \
         patch.object(admin_routes, "_get_engine_pool", lambda: pool), \
         patch.object(admin_routes, "clear_hot_cache", AsyncMock(return_value={})) as clear:
        req = admin_routes.CacheExperimentRequest(**body)
        result = asyncio.run(admin_routes.post_cache_experiment(req, is_admin=True))
    return result, sched, pool, clear


class TestRoute:
    def test_disabled_returns_404(self, monkeypatch):
        monkeypatch.delenv(ce.ENV_FLAG, raising=False)
        with pytest.raises(HTTPException) as exc:
            _post({"hot_cache_max_size": "2GB"})
        assert exc.value.status_code == 404

    def test_applies_size_and_resets(self, monkeypatch):
        monkeypatch.setenv(ce.ENV_FLAG, "1")
        result, sched, pool, clear = _post({"hot_cache_max_size": "2GB"})
        assert sched.paged_ssd_cache_manager._hot_cache_max_bytes == 2 * 1024**3
        assert pool._scheduler_config.hot_cache_budget.max_bytes == 2 * 1024**3
        sched.block_aware_cache.clear.assert_called_once()
        clear.assert_awaited_once()
        assert result["cleared"]["m"]["prefix_entries"] == 7

    @pytest.mark.parametrize("size", ["0", "auto", "-1GB", "lots"])
    def test_rejects_bad_sizes(self, monkeypatch, size):
        monkeypatch.setenv(ce.ENV_FLAG, "1")
        with pytest.raises(HTTPException) as exc:
            _post({"hot_cache_max_size": size})
        assert exc.value.status_code == 400

    def test_size_change_requires_reset(self, monkeypatch):
        monkeypatch.setenv(ce.ENV_FLAG, "1")
        with pytest.raises(HTTPException) as exc:
            _post({"hot_cache_max_size": "2GB", "reset": False})
        assert exc.value.status_code == 400

    def test_busy_returns_409_and_changes_nothing(self, monkeypatch):
        monkeypatch.setenv(ce.ENV_FLAG, "1")
        sched = _scheduler(running={"r": 1})
        with pytest.raises(HTTPException) as exc:
            _post({"hot_cache_max_size": "2GB", "idle_timeout_s": 0}, sched=sched)
        assert exc.value.status_code == 409
        sched.block_aware_cache.clear.assert_not_called()
        assert sched.paged_ssd_cache_manager._hot_cache_max_bytes == 8 << 30

    def test_unknown_fields_rejected(self):
        # Settings not implemented yet must fail loudly, not be ignored.
        with pytest.raises(Exception):
            admin_routes.CacheExperimentRequest(snapshot_every=4)
