"""Dev-only cache experiment controls (enabled by OMLX_CACHE_EXPERIMENTS=1).

Lets a benchmark driver switch hot-cache settings on the loaded models and
reset every piece of prefix-cache state between runs, without the model
unload/reload that the normal cache-settings path performs. Not part of the
supported admin API.

A reset clears more than the hot tier: the benchmark replays byte-identical
prompts, so block hashes repeat across runs, and any surviving tip-lineage /
tail / prefix-index state from a previous run would change supersession
behavior in the next one.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable

logger = logging.getLogger(__name__)

ENV_FLAG = "OMLX_CACHE_EXPERIMENTS"


def experiments_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def scheduler_busy(scheduler: Any) -> str | None:
    """Return why ``scheduler`` is not idle, or None when it is idle."""
    for attr in ("waiting", "prefilling", "running"):
        if getattr(scheduler, attr, None):
            return f"{attr} requests present"
    inflight = getattr(scheduler, "_inflight_store_futures", None) or {}
    if any(not f.done() for f in list(inflight.values())):
        return "async cache stores in flight"
    return None


def wait_until_idle(
    schedulers: Iterable[tuple[str, Any]], timeout_s: float = 30.0
) -> dict[str, str]:
    """Poll until every scheduler is idle; return {model_id: reason} still busy."""
    deadline = time.monotonic() + timeout_s
    schedulers = list(schedulers)
    while True:
        busy = {
            model_id: reason
            for model_id, sched in schedulers
            if (reason := scheduler_busy(sched)) is not None
        }
        if not busy or time.monotonic() >= deadline:
            return busy
        time.sleep(0.1)


def reset_prefix_state(scheduler: Any) -> dict[str, int]:
    """Clear prefix index, lineage sets, paged block metadata and hot entries."""
    out = {"prefix_entries": 0, "hot_entries": 0}
    prefix_cache = getattr(scheduler, "block_aware_cache", None)
    if prefix_cache is not None and hasattr(prefix_cache, "clear"):
        out["prefix_entries"] = int(prefix_cache.clear() or 0)
    ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
    if ssd_manager is not None:
        if hasattr(ssd_manager, "clear_hot_cache"):
            out["hot_entries"] = int(ssd_manager.clear_hot_cache() or 0)
        # Instrumentation counters (see get_cache_breakdown).
        if hasattr(ssd_manager, "_evict_profile"):
            ssd_manager._evict_profile = {}
        if not getattr(ssd_manager, "_hot_cache_only", True) and hasattr(ssd_manager, "clear"):
            # SSD tier on: start every run from an empty SSD cache too.
            out["ssd_files"] = int(ssd_manager.clear() or 0)
        if hasattr(ssd_manager, "reset_ssd_write_accounting"):
            ssd_manager.reset_ssd_write_accounting()
        stats = getattr(ssd_manager, "_stats", None)
        if isinstance(stats, dict):
            # Per-run counters (otherwise cumulative since server start).
            for key in ("hot_cache_evictions", "hot_cache_admission_refusals"):
                if key in stats:
                    stats[key] = 0
    for attr in ("_sidecar_boundaries_skipped", "_staging_ram_kept",
                 "_staging_ram_kept_bytes", "_staging_spilled"):
        if hasattr(scheduler, attr):
            setattr(scheduler, attr, 0)
    store = getattr(scheduler, "_boundary_snapshot_store", None)
    if store is not None and hasattr(store, "_staged_files_written"):
        with store._pending_lock:
            store._staged_files_written = 0
            store._staged_bytes_written = 0
    return out


def apply_hot_cache_max_bytes(
    pool: Any, schedulers: Iterable[tuple[str, Any]], max_bytes: int
) -> None:
    """Set the hot-tier byte limit live: pool config, shared budget, managers.

    Call only after ``reset_prefix_state`` emptied the hot tier, so no
    shrinking/eviction is needed here.
    """
    config = getattr(pool, "_scheduler_config", None)
    if config is not None:
        config.hot_cache_max_size = max_bytes
        budget = getattr(config, "hot_cache_budget", None)
        if budget is not None:
            budget.max_bytes = max_bytes
    for _model_id, scheduler in schedulers:
        sched_config = getattr(scheduler, "config", None)
        if sched_config is not None and hasattr(sched_config, "hot_cache_max_size"):
            sched_config.hot_cache_max_size = max_bytes
        ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
        if ssd_manager is not None and hasattr(ssd_manager, "_hot_cache_max_bytes"):
            ssd_manager._hot_cache_max_bytes = max_bytes


def apply_hot_cache_eviction_policy(
    pool: Any, schedulers: Iterable[tuple[str, Any]], policy: str
) -> None:
    """Set the hot-tier eviction policy live on the budget and every manager.

    Call only after ``reset_prefix_state`` emptied the hot tier: the child
    index is rebuilt from scratch on the next inserts.
    """
    config = getattr(pool, "_scheduler_config", None)
    if config is not None:
        config.hot_cache_eviction_policy = policy
        budget = getattr(config, "hot_cache_budget", None)
        if budget is not None:
            budget.eviction_policy = policy
    for _model_id, scheduler in schedulers:
        sched_config = getattr(scheduler, "config", None)
        if sched_config is not None:
            sched_config.hot_cache_eviction_policy = policy
        ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
        if ssd_manager is not None and hasattr(ssd_manager, "_hot_eviction_policy"):
            ssd_manager._hot_eviction_policy = policy
            ssd_manager._hot_children = {}


def apply_gdn_snapshot_interval(
    pool: Any, schedulers: Iterable[tuple[str, Any]], interval: int
) -> None:
    """Set gdn_snapshot_interval live on the pool template and prefix caches."""
    config = getattr(pool, "_scheduler_config", None)
    if config is not None:
        config.gdn_snapshot_interval = interval
    for _model_id, scheduler in schedulers:
        sched_config = getattr(scheduler, "config", None)
        if sched_config is not None:
            sched_config.gdn_snapshot_interval = interval
        prefix_cache = getattr(scheduler, "block_aware_cache", None)
        if prefix_cache is not None and hasattr(prefix_cache, "_gdn_snapshot_interval"):
            prefix_cache._gdn_snapshot_interval = interval


def apply_gdn_staging_ram_budget(
    pool: Any, schedulers: Iterable[tuple[str, Any]], budget: tuple[int, float]
) -> None:
    """Set the RAM-first staging budget live (pool template and schedulers).

    ``budget`` is (bytes, fraction of the hot cache) from
    ``parse_gdn_staging_ram_budget``.
    """
    budget_bytes, fraction = budget
    configs = [getattr(pool, "_scheduler_config", None)]
    configs += [getattr(s, "config", None) for _m, s in schedulers]
    for config in configs:
        if config is not None:
            config.gdn_staging_ram_budget_bytes = budget_bytes
            config.gdn_staging_ram_budget_fraction = fraction


def apply_ssd_cache_write_policy(
    pool: Any, schedulers: Iterable[tuple[str, Any]], policy: str | None,
    min_chain_tokens: int | None = None,
) -> None:
    """Set ssd_cache_write_policy / min chain threshold live (pool and SSD managers)."""
    config = getattr(pool, "_scheduler_config", None)
    if config is not None:
        if policy is not None:
            config.ssd_cache_write_policy = policy
        if min_chain_tokens is not None:
            config.ssd_cache_min_chain_tokens = min_chain_tokens
    for _model_id, scheduler in schedulers:
        sched_config = getattr(scheduler, "config", None)
        if sched_config is not None:
            if policy is not None:
                sched_config.ssd_cache_write_policy = policy
            if min_chain_tokens is not None:
                sched_config.ssd_cache_min_chain_tokens = min_chain_tokens
        ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
        if ssd_manager is not None and hasattr(ssd_manager, "_ssd_write_policy"):
            if policy is not None:
                ssd_manager._ssd_write_policy = policy
            if min_chain_tokens is not None:
                ssd_manager._ssd_min_chain_tokens = min_chain_tokens


def current_settings(pool: Any, schedulers: Iterable[tuple[str, Any]]) -> dict:
    config = getattr(pool, "_scheduler_config", None)
    budget = getattr(config, "hot_cache_budget", None) if config is not None else None
    models = {}
    for model_id, scheduler in schedulers:
        ssd_manager = getattr(scheduler, "paged_ssd_cache_manager", None)
        models[model_id] = {
            "hot_cache_max_bytes": getattr(ssd_manager, "_hot_cache_max_bytes", None),
            "hot_cache_bytes": getattr(ssd_manager, "_hot_cache_total_bytes", None),
            "hot_cache_eviction_policy": getattr(ssd_manager, "_hot_eviction_policy", None),
            "gdn_snapshot_interval": getattr(
                getattr(scheduler, "block_aware_cache", None), "_gdn_snapshot_interval", None
            ),
            "gdn_snapshots_thinned": getattr(
                getattr(scheduler, "block_aware_cache", None), "_gdn_snapshots_thinned", None
            ),
            "ssd_cache_write_policy": getattr(ssd_manager, "_ssd_write_policy", None),
            "gdn_sidecars_thinned": getattr(
                getattr(scheduler, "block_aware_cache", None), "_gdn_sidecars_thinned", None
            ),
            "sidecar_boundaries_skipped": getattr(scheduler, "_sidecar_boundaries_skipped", None),
            "staging_ram_budget_bytes": (
                scheduler._staging_ram_budget_bytes()
                if callable(getattr(scheduler, "_staging_ram_budget_bytes", None))
                else None
            ),
            "staging_ram_budget_fraction": getattr(
                getattr(scheduler, "config", None), "gdn_staging_ram_budget_fraction", None
            ),
            "staging_ram_kept": getattr(scheduler, "_staging_ram_kept", None),
            "staging_ram_kept_bytes": getattr(scheduler, "_staging_ram_kept_bytes", None),
            "staging_spilled": getattr(scheduler, "_staging_spilled", None),

            "hot_cache_only": getattr(ssd_manager, "_hot_cache_only", None),
            "ssd": (
                ssd_manager.get_ssd_write_report()
                if hasattr(ssd_manager, "get_ssd_write_report") else None
            ),
            "boundary_staging_written": getattr(
                getattr(scheduler, "_boundary_snapshot_store", None), "staged_write_stats", None
            ),
            "block_size": getattr(getattr(scheduler, "config", None), "paged_cache_block_size", None),
        }
    return {
        "budget_max_bytes": getattr(budget, "max_bytes", None),
        "budget_total_bytes": getattr(budget, "total_bytes", None),
        "budget_eviction_policy": getattr(budget, "eviction_policy", None),
        "models": models,
    }
