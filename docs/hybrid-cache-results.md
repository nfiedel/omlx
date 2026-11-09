# Hybrid cache results

Measured effect of four small cache changes on a hybrid (attention + GDN) model: hit rate with a fixed RAM tier, SSD writes and reads with the SSD tier on, and the costs.

Mac Studio M5 Ultra, 256 GB · Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp, MTP on · upstream base `2165e757` · 2026-11-08

## Summary

| | Before → after |
|---|---|
| Hit ratio, 8 GB hot tier, one conversation 92k → 121k tokens (SSD off) | 33% → **86%** |
| Time to first token per turn, same run | 33–40 s → **2.3 s** |
| SSD written, same conversation, SSD on, 6 GB hot tier | 19.3 → **5.5 GB** (SSD read 62.5 → **0 GB**) |
| SSD written, agent + 6 sub-agents, all four changes (default S=16k), same hit ratio | 34.4 → **8.4 GB** |

## Changes

Each block of this model is about 57 MB of attention KV plus a 116 MB recurrent-state (GDN) snapshot per 2,048 tokens. A restore needs only the snapshot where it resumes. All settings are opt-in except #3.

| # | Setting | What it does | Diff |
|---|---|---|---|
| 0 | bug fix | Block rewrites (rotating tip strip, dedup backfill) keep `parent_hash`. | [commit](https://github.com/nfiedel/omlx/commit/c90ea449) |
| 1 | `hot_cache_eviction_policy: "chain_aware_lru"` | Evict only blocks nothing else builds on, orphans first; refuse admission rather than break a chain. LRU otherwise evicts a chain's first block first and strands the rest. | [compare](https://github.com/nfiedel/omlx/compare/pr/0-rewrite-keeps-parent-hash...pr/1-chain-aware-lru) |
| 2 | `gdn_snapshot_interval: N` | Keep the GDN snapshot only every N-th block plus each conversation's newest block; restores walk back to it. Prefill captures only those boundaries. | [compare](https://github.com/nfiedel/omlx/compare/pr/1-chain-aware-lru...pr/2-gdn-snapshot-interval) |
| 3 | `gdn_staging_ram_budget` (default `"30%"`) | Keep prefill snapshots in RAM up to a share of the hot cache instead of staging them to SSD; spill beyond it. Embedded layout with SSD on. | [compare](https://github.com/nfiedel/omlx/compare/pr/2-gdn-snapshot-interval...pr/3-gdn-staging-ram-budget) |
| 4 | `ssd_cache_write_policy: "min_chain_tokens"` | Write a block leaving RAM only if its conversation reached S tokens (default 16k) or a block built on it was written. Requires #1. | [compare](https://github.com/nfiedel/omlx/compare/pr/3-gdn-staging-ram-budget...pr/4-ssd-min-chain-write-policy) |

## Hot tier only (SSD off)

Once one conversation is larger than the hot tier, LRU evicts its first block on every store and later turns find nothing. #1 keeps the conversation's prefix; #2 makes each block about half the size, so twice as much fits.

**8 GB hot tier, one conversation 92k → 121k tokens**

| | Today (lru, N=1) | #1 + #2 (N=4) |
|---|---:|---:|
| Hit ratio | 33.4% | **85.8%** |
| Tokens prefilled | 569k | **121k** (−79%) |
| Time to first token after the first turn | 33–40 s from 104.6k on | **2.2–2.4 s every turn** |
| Wall time | 191 s | **48 s** |

**2 GB hot tier: hit ratio / tokens prefilled** (a smaller tier, so the effect shows on shorter conversations; "–" = not run)

| Workload | Today | #1 only | #2 only (N=4) | #1 + #2 |
|---|---:|---:|---:|---:|
| One agent → 62k tokens | 10.2% / 433k | 56.1% / 212k | 53.8% / 223k | **84.4% / 75k** |
| Two interleaved agents → 37k each | 16.6% / 289k | 36.1% / 221k | – | **66.4% / 116k** |
| Agent + side calls → 50k | 32.1% / 270k | 67.2% / 130k | – | **85.1% / 59k** |

## SSD on: #1 + #2 + #3

With SSD on, evicted blocks go to SSD and come back, so hit ratios match across configurations; the differences are SSD traffic and latency. Compared with the same embedded layout without changes, at the same RAM size.

| Workload (hot tier) | Without: written / read | With #1–#3: written / read | Writes | Hit ratio |
|---|---:|---:|---:|---:|
| One agent → 62k tokens (3 GB) | 11.4 / 21.5 GB | **3.0 / 0 GB** | −74% | 87.2% → 87.2% |
| Agent + side calls → 50k (3 GB) | 11.8 / 11.7 GB | **3.9 / 0 GB** | −67% | 86.2% → 86.2% |
| Regenerates and branches → 22k (3 GB) | 6.3 / 0 GB | **2.3 / 0 GB** | −63% | 86.9% → 81.7% |
| One conversation 92k → 121k (6 GB) | 19.3 / 62.5 GB | **5.5 / 0 GB** | −72% | 85.8% → 85.8% |

SSD written includes a final flush of the hot tier, as at shutdown. With #1–#3, almost all remaining writes are that one-time flush; during the runs only ~0.13 GB was written.

- **#1** leaves writes unchanged and cuts SSD reads 71–78% when RAM is smaller than the conversation, because the conversation stops evicting its own prefix.
- **#2** cuts writes 33% (N=2) to 50% (N=8) on its own; at N=4, −45%.
- **#3** cuts writes 43–46% on its own by removing all prefill staging writes (2.9–5.2 GB per run).
- **#2 and #3 compound:** #2 shrinks what is captured and stored, #3 keeps it off the disk until a block leaves RAM.

## SSD write filter: #4 on top of #1–#3

Short, transient conversations such as side calls and sub-agents are rarely reused once they leave RAM. Those still in use stay in RAM under LRU order, and one that is evicted is cheap to recompute. #4 keeps them off the SSD; every block it writes keeps its parent on SSD, so what is written stays usable.

| Workload (3 GB) | Write all | S = 16k | S = 32k | Cost |
|---|---:|---:|---:|---|
| Main agent + 6 sub-agents (11k–40k tokens each) | 10.2 GB | 8.4 GB (−18%) | **4.2 GB (−59%)** | +1.7% prefill at every S |
| Agent + side calls | 3.9 GB | **2.3 GB (−40%)** | **2.3 GB (−40%)** | none |
| One agent → 62k tokens | 3.0 GB | 2.8 GB (−8%) | – | none |

The sub-agent cost is one dropped block: the partial last block of the shared sub-agent system prompt, which nothing builds on. Two sub-agents re-prefilled it (+3.1k tokens, about +0.9 s on two turns). Hit ratio 80.3% → 80.0%.

**All four vs no changes**

| Workload (3 GB) | No changes | #1–#3 | #1–#4, S = 16k | #1–#4, S = 32k | Hit ratio |
|---|---:|---:|---:|---:|---:|
| Main agent + 6 sub-agents | 34.4 GB | 10.2 GB (−70%) | **8.4 GB (−76%)** | **4.2 GB (−88%)** | 80.5% → 80.0% |
| Agent + side calls | 11.8 GB | 3.9 GB (−67%) | **2.3 GB (−80%)** | **2.3 GB (−80%)** | 86.2% → 86.2% |
| One agent → 62k tokens | 11.4 GB | 3.0 GB (−74%) | **2.8 GB (−76%)** | – | 87.2% → 87.2% |

SSD written including the final flush; "no changes" includes 15.3 GB of prefill staging on the sub-agent workload. The #1–#4 runs used a 2 GB staging budget; nothing spilled, so the 30% default gives the same result at this size.

## SSD wear

SSD bytes written per newly prefilled token, from the runs above. Reads do not wear the drive.

| Configuration | KB written per prefilled token |
|---|---:|
| Today's default with SSD on (sidecar layout) | ~100 (65–80 measured GDN staging + ~28 KV, estimated) |
| Embedded layout without changes | 159–184 |
| **#1 + #2 + #3** | **45–49** |

| Lifespan of a 600 TBW 1 TB drive | Today's default | #1 + #2 + #3 |
|---|---:|---:|
| Heavy agentic use: 65M new tokens/day (25% prefill duty at ~3,000 tok/s) | ~3 months | ~6.5 months |
| 10M new tokens/day | ~1.6 years | ~3.5 years |

Estimate. Apple does not publish TBW for Mac internal SSDs; 600 TBW is typical for a 1 TB consumer TLC drive. #4 lowers writes further on side-call and sub-agent workloads.

## Costs

Append-only agent workloads pay nothing. Branching mid-conversation re-prefills up to N−1 blocks back to the last snapshot; regenerating the last turn stays free.

| Regenerates and branches (3 GB, SSD on) | N=1 | N=2 | N=3 | N=4 | N=8 |
|---|---:|---:|---:|---:|---:|
| Hit ratio | 86.9% | 83.5% | 82.6% | 81.7% | 71.3% |
| p95 time to first token | 1.28 s | 1.97 s | 2.47 s | 2.93 s | 4.23 s |

N=4 is the suggested setting for agents. N=8 is too coarse for interactive editing.

#3 holds prefill snapshots in RAM only during prefill, capped at 30% of the hot cache by default; anything over spills to SSD as today. #4's cost is bounded: a dropped conversation re-prefills at most S tokens.

## Correctness

- **Real model:** 8 prompts (8k–62k tokens), greedy decoding with logprobs, edit and append cases. Thinned restores at N = 2, 3, 4 and 8 are bit-identical to N=1 in tokens and logprobs (max |Δ| = 0.0) across 80 generations, including restores that walked back 5 blocks. A second check with #1–#3 all on was bit-identical too.
- **Unit tests:** the existing prefix-cache, boundary, SSD, hot-cache and settings suites pass on every branch, plus new tests: seeded random operation sequences for the eviction invariants and for SSD contents staying usable under #4, and bit-identical restored tensors at every walk-back point.

## Method and limits

- Deterministic replays of scripted multi-turn agent conversations (seed 1234), byte-identical across runs, with `scripts/cache_bench_replay.py`. Each run starts with empty RAM and SSD caches. Token and byte counts are exact; timings are single runs and vary ~2%. Correctness checks use `scripts/cache_correctness_check.py`.
- The SSD-on comparisons use the embedded GDN layout for both sides. Today's default with SSD on is the sidecar layout; the wear row for it is partly estimated, and a like-for-like run is still to do.
- One hybrid model so far. #4 was measured only in the embedded layout.

*AI-assisted (Claude Code); reviewed and validated by the author.*
