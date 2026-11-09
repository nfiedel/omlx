# SPDX-License-Identifier: Apache-2.0
"""Tests for ssd_cache_write_policy (default "all", and "min_chain_tokens").

With the SSD tier on, every dirty hot entry leaving the hot tier is written to
SSD ("all"). "min_chain_tokens" writes a block only if its chain reached
ssd_cache_min_chain_tokens, or a block built on it was written, so short
transient chains never reach SSD. The written set must stay prefix-closed per
chain, because SSD content is only usable as an unbroken run from a chain's
first block.
"""

from __future__ import annotations

import random

import pytest

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager, SharedHotCacheBudget
from omlx.settings import GlobalSettings

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

TOKENS = 16


def _data():
    return [(mx.zeros((1, 2, TOKENS, 16)), mx.zeros((1, 2, TOKENS, 16))) for _ in range(2)]


def _block_bytes(tmp_path) -> int:
    probe = PagedSSDCacheManager(
        cache_dir=tmp_path / "probe", max_size_bytes=1 << 30, hot_cache_max_bytes=1 << 30
    )
    probe.save_block(b"probe", _data(), TOKENS, model_name="m", layer_cache_types=["KVCache"] * 2)
    size = probe._hot_cache_total_bytes
    probe.close()
    return size


def _manager(tmp_path, policy, cap_blocks, *, eviction="lru", write_through=False, budget=False, name="ssd",
             min_chain=16384):
    cap = int((cap_blocks + 0.5) * _block_bytes(tmp_path))
    return PagedSSDCacheManager(
        ssd_cache_min_chain_tokens=min_chain,
        cache_dir=tmp_path / name,
        max_size_bytes=1 << 30,
        hot_cache_max_bytes=cap,
        hot_cache_write_through=write_through,
        hot_cache_budget=SharedHotCacheBudget(cap, eviction_policy=eviction) if budget else None,
        hot_cache_eviction_policy=eviction,
        ssd_cache_write_policy=policy,
    )


def _save(mgr, key: bytes, parent: bytes | None = None, end: int = 0):
    assert mgr.save_block(
        key, _data(), TOKENS, model_name="m", layer_cache_types=["KVCache"] * 2, parent_hash=parent,
        chain_end_tokens=end,
    )


def _on_ssd(mgr, key: bytes) -> bool:
    return mgr._index.contains(key)


@pytest.fixture(params=[False, True], ids=["local", "budget"])
def budget(request):
    return request.param


class TestRule:
    def test_all_writes_everything(self, tmp_path, budget):
        mgr = _manager(tmp_path, "all", 2, budget=budget)
        for k in (b"a", b"b", b"c"):
            _save(mgr, k)
        assert _on_ssd(mgr, b"a")  # evicted, never read, still written
        mgr.close()

    def test_short_chain_dropped_and_regret_counted(self, tmp_path, budget):
        mgr = _manager(tmp_path, "min_chain_tokens", 2, eviction="chain_aware_lru", budget=budget,
                       min_chain=16384)
        for k in (b"a", b"b", b"c"):
            _save(mgr, k, end=4096)  # three separate short chains
        assert not _on_ssd(mgr, b"a")
        assert mgr._stats["ssd_writes_skipped_short"] == 1
        assert mgr.load_block_with_metadata(b"a") == (None, None)
        # A restore that stops at the dropped block reports the tokens it lost.
        mgr.note_restore_loss(b"a", 3 * TOKENS)
        assert mgr._stats["ssd_write_regret"] == 1
        assert mgr._stats["ssd_write_regret_tokens"] == 3 * TOKENS
        mgr.note_restore_loss(b"a", TOKENS)  # counted once per dropped block
        assert mgr._stats["ssd_write_regret"] == 1
        mgr.note_restore_loss(b"never-dropped", TOKENS)  # ordinary misses are not regret
        assert mgr._stats["ssd_write_regret"] == 1
        mgr.close()

    def test_orphan_dropped_even_if_long(self, tmp_path, budget):
        mgr = _manager(tmp_path, "min_chain_tokens", 3, eviction="chain_aware_lru", budget=budget,
                       min_chain=16384)
        _save(mgr, b"p", end=16384)
        _save(mgr, b"c", parent=b"p", end=20480)
        mgr._hot_cache_remove(b"p")  # parent gone from RAM and never on SSD
        for k in (b"x", b"y", b"z", b"w"):
            _save(mgr, k, end=4096)
        assert not _on_ssd(mgr, b"c")
        assert mgr._stats["ssd_writes_skipped_orphan"] >= 1
        mgr.close()

    def test_refused_overflow_is_written_under_all(self, tmp_path):
        mgr = _manager(tmp_path, "all", 3, eviction="chain_aware_lru", budget=True)
        parent = None
        keys = [f"blk{i}".encode() for i in range(6)]
        for k in keys:
            _save(mgr, k, parent)
            parent = k
        for k in keys[3:]:
            assert _on_ssd(mgr, k), k
        mgr.close()

    def test_write_through_unaffected(self, tmp_path):
        mgr = _manager(tmp_path, "min_chain_tokens", 2, eviction="chain_aware_lru", write_through=True)
        for k in (b"a", b"b", b"c"):
            _save(mgr, k, end=4096)
        assert all(_on_ssd(mgr, k) for k in (b"a", b"b", b"c"))
        mgr.close()


class TestFlushes:
    def test_shutdown_follows_policy(self, tmp_path):
        mgr = _manager(tmp_path, "min_chain_tokens", 10, eviction="chain_aware_lru", min_chain=16384)
        _save(mgr, b"long0", end=8192)
        _save(mgr, b"long1", parent=b"long0", end=16384)
        _save(mgr, b"short", end=4096)
        mgr.close()
        assert mgr._stats["ssd_writes_skipped_short"] == 1
        reopened = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd", max_size_bytes=1 << 30, hot_cache_max_bytes=1 << 20
        )
        assert reopened.has_block(b"long0") and reopened.has_block(b"long1")
        assert not reopened.has_block(b"short")
        reopened.close()

    def test_shutdown_all_writes_everything(self, tmp_path):
        mgr = _manager(tmp_path, "all", 4)
        _save(mgr, b"x")
        mgr.close()
        reopened = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd", max_size_bytes=1 << 30, hot_cache_max_bytes=1 << 20
        )
        assert reopened.has_block(b"x")
        reopened.close()


class TestConfig:
    def test_invalid_policy_rejected(self, tmp_path):
        for bad in ("likely_reused", "sometimes"):
            with pytest.raises(ValueError):
                PagedSSDCacheManager(
                    cache_dir=tmp_path / "x", max_size_bytes=1 << 20,
                    hot_cache_max_bytes=1 << 20, ssd_cache_write_policy=bad,
                )

    def test_settings_roundtrip_and_validation(self, tmp_path):
        s = GlobalSettings(base_path=tmp_path)
        assert s.cache.ssd_cache_write_policy == "all"
        assert s.cache.ssd_cache_min_chain_tokens == 16384
        s.cache.hot_cache_eviction_policy = "chain_aware_lru"
        s.cache.ssd_cache_write_policy = "min_chain_tokens"
        s.cache.ssd_cache_min_chain_tokens = 32768
        s.save()
        loaded = GlobalSettings.load(base_path=tmp_path)
        assert loaded.cache.ssd_cache_write_policy == "min_chain_tokens"
        config = loaded.to_scheduler_config()
        assert config.ssd_cache_write_policy == "min_chain_tokens"
        assert config.ssd_cache_min_chain_tokens == 32768
        assert not [e for e in loaded.validate() if "ssd_cache" in e]
        loaded.cache.ssd_cache_write_policy = "sometimes"
        assert any("ssd_cache_write_policy" in e for e in loaded.validate())


# --- ssd_cache_write_policy = "min_chain_tokens" ------------------------------------


def _chain(mgr, prefix: str, n_blocks: int, block_tokens: int = 4096, root_parent=None):
    keys, parent = [], root_parent
    for i in range(n_blocks):
        key = f"{prefix}{i}".encode()
        _save(mgr, key, parent, end=(i + 1) * block_tokens)
        keys.append(key)
        parent = key
    return keys


class TestMinChainTokens:
    def test_flush_writes_long_chain_with_all_ancestors_and_drops_short(self, tmp_path):
        mgr = _manager(tmp_path, "min_chain_tokens", 50, eviction="chain_aware_lru", min_chain=16384)
        long_keys = _chain(mgr, "L", 6)   # ends 4k..24k: reaches the threshold
        short_keys = _chain(mgr, "S", 3)  # ends 4k..12k: never does
        mgr.clear_hot_cache()
        assert all(_on_ssd(mgr, k) for k in long_keys)  # incl. ancestors below 16k
        assert not any(_on_ssd(mgr, k) for k in short_keys)
        assert mgr._stats["ssd_writes_skipped_short"] == 3
        mgr.close()

    def test_shared_prefix_written_through_long_child(self, tmp_path):
        mgr = _manager(tmp_path, "min_chain_tokens", 50, eviction="chain_aware_lru", min_chain=16384)
        sys_keys = _chain(mgr, "SYS", 2)                       # shared prefix, ends 4k, 8k
        _chain(mgr, "SIDE", 1, root_parent=sys_keys[-1])       # short side call
        long_keys = [f"M{i}".encode() for i in range(3)]
        parent = sys_keys[-1]
        for i, k in enumerate(long_keys):
            _save(mgr, k, parent, end=8192 + (i + 1) * 4096)  # ends 12k, 16k, 20k
            parent = k
        mgr.clear_hot_cache()
        assert all(_on_ssd(mgr, k) for k in sys_keys + long_keys)
        assert not _on_ssd(mgr, b"SIDE0")
        mgr.close()

    def test_eviction_path_keeps_runs_complete(self, tmp_path):
        """Short chain evicted (dropped); a long chain overflowing RAM is written
        newest-first and its ancestors follow when they leave RAM."""
        mgr = _manager(tmp_path, "min_chain_tokens", 4, eviction="chain_aware_lru", budget=True,
                       min_chain=16384)
        short_keys = _chain(mgr, "S", 2)
        long_keys = _chain(mgr, "L", 8)  # ends up to 32k; overflows a 4-block tier
        mgr.close()  # shutdown flush follows the policy too
        on_ssd = {h for h in mgr._index.get_all_hashes()}
        assert set(long_keys) <= on_ssd
        assert not (set(short_keys) & on_ssd)
        for h in on_ssd:
            parent = getattr(mgr._index.get(h), "parent_hash", None)
            assert parent is None or parent in on_ssd

    def test_falls_back_to_all_without_chain_aware_lru(self, tmp_path):
        mgr = _manager(tmp_path, "min_chain_tokens", 50, eviction="lru")
        keys = _chain(mgr, "S", 2)
        mgr.clear_hot_cache()
        assert all(_on_ssd(mgr, k) for k in keys)
        mgr.close()

    def test_settings_validation(self, tmp_path):
        s = GlobalSettings(base_path=tmp_path)
        s.cache.ssd_cache_write_policy = "min_chain_tokens"
        s.cache.hot_cache_eviction_policy = "lru"
        assert any("requires" in e for e in s.validate())
        s.cache.hot_cache_eviction_policy = "chain_aware_lru"
        assert not [e for e in s.validate() if "min_chain" in e or "requires" in e]
        s.cache.ssd_cache_min_chain_tokens = 0
        assert any("ssd_cache_min_chain_tokens" in e for e in s.validate())


@pytest.mark.parametrize("min_chain", [4096 * 2, 4096 * 4])
def test_min_chain_ssd_contents_stay_prefix_closed(tmp_path, min_chain, budget):
    """Seeded random conversations under min_chain_tokens + chain_aware_lru."""
    for seed in range(10):
        rng = random.Random(seed)
        mgr = _manager(tmp_path, "min_chain_tokens", 5, eviction="chain_aware_lru", budget=budget,
                       name=f"mc{seed}", min_chain=min_chain)
        chains: list[list[bytes]] = []
        ends: dict[bytes, int] = {}
        for step in range(60):
            if not chains or rng.random() < 0.3:
                chain = []
                chains.append(chain)
            else:
                chain = rng.choice(chains)
                for i, k in enumerate(chain):  # restore; re-store from the first miss
                    if mgr.load_block(k) is None:
                        for j in range(i, len(chain)):
                            _save(mgr, chain[j], chain[j - 1] if j else None, end=ends[chain[j]])
                        break
            key = f"mc{seed}-{step}".encode()
            ends[key] = (len(chain) + 1) * 4096
            _save(mgr, key, chain[-1] if chain else None, end=ends[key])
            chain.append(key)
        mgr.close()
        on_ssd = set(mgr._index.get_all_hashes())
        for h in on_ssd:
            parent = getattr(mgr._index.get(h), "parent_hash", None)
            assert parent is None or parent in on_ssd, f"seed={seed}: {h!r} without parent"
