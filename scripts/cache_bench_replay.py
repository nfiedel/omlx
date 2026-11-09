#!/usr/bin/env python3
"""Deterministic prefix-cache benchmark: replay scripted multi-turn workloads.

Every request sequence is generated up front from a seed and the bundled
benchmark corpora. Model output is never fed back into later prompts, so each
cache configuration sees byte-identical requests and cache metrics (cached /
prefilled tokens) are directly comparable between runs. Decode is kept tiny
(``max_tokens``) because the benchmark measures prefill and cache behavior.

Requires a running oMLX server started with OMLX_CACHE_EXPERIMENTS=1 (each run
begins with POST /admin/api/cache-experiment, which resets all prefix-cache
state and applies the hot-cache size without reloading the model).

Example:
    OMLX_KEY=... .venv/bin/python scripts/cache_bench_replay.py \\
        --model Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp \\
        --tokenizer /path/to/model/dir --workload W1 --hot-cache 2GB \\
        --label lru-n1 --out ../bench-results
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import httpx

CORPUS_DIR = Path(__file__).resolve().parent.parent / "omlx" / "admin" / "bench_corpora"
DEFAULT_CORPORA = ("code_python.txt", "code_mixed.txt")


# --------------------------------------------------------------------------
# Content: exact-size token chunks from the corpora
# --------------------------------------------------------------------------


class ChunkSource:
    """Hands out consecutive, non-overlapping text chunks of ~N tokens."""

    def __init__(self, tokenizer: Any, corpora: tuple[str, ...], seed: int):
        ids: list[int] = []
        for name in corpora:
            ids.extend(tokenizer((CORPUS_DIR / name).read_text())["input_ids"])
        self._tok = tokenizer
        self._ids = ids
        # Seeded start offset so different seeds give different content.
        self._pos = random.Random(seed).randrange(0, len(ids) // 4)

    def take(self, n_tokens: int) -> str:
        if self._pos + n_tokens > len(self._ids):
            self._pos = 0  # wrap; content repeats only for very long workloads
        piece = self._ids[self._pos : self._pos + n_tokens]
        self._pos += n_tokens
        return self._tok.decode(piece)


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Req:
    conv: str  # conversation id, for reporting
    kind: str  # "turn" | "side" | "regenerate" | "branch-K"
    messages: list[dict]


class Conversation:
    """A scripted conversation: system prompt + alternating user/assistant."""

    def __init__(self, cid: str, system: str, src: ChunkSource):
        self.cid = cid
        self.src = src
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.turns = 0

    def next_turn(self, add_tokens: int) -> Req:
        """Append a user message of ~add_tokens; return the request for it."""
        if self.turns > 0:
            # Scripted reply for the previous turn (never the model's output).
            self.messages.append(self.reply(self.turns))
        self.turns += 1
        self.messages.append(
            {
                "role": "user",
                "content": f"[turn {self.turns}] Tool output:\n" + self.src.take(add_tokens),
            }
        )
        return Req(self.cid, "turn", list(self.messages))

    @staticmethod
    def reply(turn: int) -> dict:
        return {"role": "assistant", "content": f"Step {turn} done. Continuing with the next file."}

    def regenerate(self) -> Req:
        """Resend the current prompt unchanged (regenerate the last reply)."""
        return Req(self.cid, "regenerate", list(self.messages))

    def branch(self, turns_back: int, add_tokens: int) -> Req:
        """Probe: keep history up to `turns_back` turns ago, then a new user msg.

        The probe does not change the conversation; the main line continues.
        """
        # messages = [system, u1, a1, u2, a2, ..., uT]: u_k at 2k-1, a_k at 2k.
        # Keep everything through turn keep_turn's (scripted) reply, so the
        # probe diverges exactly at the new user message.
        keep_turn = max(1, self.turns - turns_back)
        prefix = self.messages[: 2 * keep_turn] + [self.reply(keep_turn)]
        prefix.append(
            {"role": "user", "content": "[branch] Different approach:\n" + self.src.take(add_tokens)}
        )
        return Req(self.cid, f"branch-{turns_back}", prefix)


def workload(name: str, src: ChunkSource, seed: int = 1234) -> Iterator[Req]:
    """Yield the request sequence for workload `name` (see DESIGN-cache-eviction.md)."""
    k = 1024
    rng = random.Random(seed)
    system8 = "You are a coding agent.\n" + src.take(8 * k)
    if name == "W1":  # single agent: 8k -> ~60k, +4k per turn
        c = Conversation("A", system8, src)
        for _ in range(13):
            yield c.next_turn(4 * k)
    elif name == "W2":  # two interleaved agents, each 8k -> ~36k
        a = Conversation("A", system8, src)
        b = Conversation("B", system8, src)
        for _ in range(7):
            yield a.next_turn(4 * k)
            yield b.next_turn(4 * k)
    elif name == "W3":  # agent + one unrelated ~500-token side call per turn
        c = Conversation("A", system8, src)
        for i in range(10):
            yield c.next_turn(4 * k)
            side = Conversation(f"side{i}", system8, src)
            yield Req(side.cid, "side", side.next_turn(500).messages)
    elif name == "W4":  # rewinds within capacity: +2k per turn to ~22k
        c = Conversation("A", system8, src)
        probes = [
            lambda: c.regenerate(),
            lambda: c.branch(2, rng.randint(600, 1400)),
            lambda: c.branch(4, rng.randint(600, 1400)),
        ]
        p = 0
        for t in range(1, 8):
            # Seeded, uneven turn sizes so branch points land at varied
            # offsets relative to the 2048-token block grid.
            yield c.next_turn(rng.randint(1300, 2700))
            if t % 2 == 0 or t == 7:
                for _ in range(2):
                    yield probes[p % len(probes)]()
                    p += 1
    elif name == "W5":  # a few short chats (home Open WebUI-like), round-robin
        system4 = "You are a helpful assistant.\n" + src.take(4 * k)
        chats = [Conversation(f"chat{i}", system4, src) for i in range(3)]
        lengths = [2, 4, 6]
        for t in range(max(lengths)):
            for chat, n in zip(chats, lengths):
                if t < n:
                    yield chat.next_turn(1 * k)
    elif name == "W6":  # main agent + transient sub-agents of varied length
        c = Conversation("A", system8, src)
        sub_system = "You are a research sub-agent. Report findings briefly.\n" + src.take(3 * k)
        targets = [10, 14, 18, 22, 28, 36]  # sub-agent final lengths (k tokens)
        for t in range(1, 13):
            yield c.next_turn(4 * k)
            if t % 2 == 0:
                target = targets[t // 2 - 1]
                sub = Conversation(f"sub{t // 2}", sub_system, src)
                turns = max(1, -(-(target - 3) // 4))  # ceil((target - 3k) / 4k)
                for _ in range(turns):
                    yield sub.next_turn(4 * k)
    elif name == "W1-8G":  # real-size cliff: 90k start, +4k x 7 (~118k)
        c = Conversation("A", system8, src)
        yield c.next_turn(82 * k)
        for _ in range(7):
            yield c.next_turn(4 * k)
    else:
        raise SystemExit(f"unknown workload {name!r}")


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


def stream_chat(client: httpx.Client, model: str, req: Req, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": req.messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    usage = None
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "prefilled_tokens": max(prompt - cached, 0),
        "ttft_s": round(ttft if ttft is not None else total, 3),
        "total_s": round(total, 3),
        "usage_present": bool(usage),
    }


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, help="model directory (for exact chunk sizes)")
    ap.add_argument("--workload", required=True, choices=["W1", "W2", "W3", "W4", "W5", "W6", "W1-8G"])
    ap.add_argument("--hot-cache", required=True, help='hot tier size for this run, e.g. "2GB"')
    ap.add_argument("--label", required=True, help="config label, e.g. lru-n1")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra cache-experiment field (e.g. gdn_snapshot_interval=4)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--time-cap", type=float, default=300.0, help="abort the run after N seconds")
    ap.add_argument("--out", default="bench-results")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the untimed warmup request (reset -> warmup -> reset)")
    ap.add_argument("--note", default="", help="short description for the run log")
    ap.add_argument("--quiet", action="store_true", help="do not print the JSON summary")
    ap.add_argument("--final-flush", action="store_true",
                    help="after the run, clear the hot tier (flushes dirty entries to SSD, "
                         "like a shutdown) and record SSD writes after the flush")
    ap.add_argument("--key-file", default=None,
                    help="file containing the API key (alternative to OMLX_KEY)")
    args = ap.parse_args()

    key = os.environ.get("OMLX_KEY")
    if not key and args.key_file:
        key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        print("set OMLX_KEY or pass --key-file", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    src = ChunkSource(tok, DEFAULT_CORPORA, args.seed)
    requests = list(workload(args.workload, src, args.seed))

    experiment: dict[str, Any] = {"hot_cache_max_size": args.hot_cache, "reset": True}
    run_name = f"{args.workload} {args.label} @ {args.hot_cache}" + (f" ({args.note})" if args.note else "")
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            experiment[k] = json.loads(v)
        except json.JSONDecodeError:
            experiment[k] = v

    out_dir = Path(args.out) / f"{args.workload}__{args.label}__{args.hot_cache}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = Path(args.out) / "run-log.txt"

    def log_run(event: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {event}: {run_name}"
        print(line, flush=True)
        with run_log.open("a") as f:
            f.write(line + "\n")

    log_run("Starting run")
    headers = {"Authorization": f"Bearer {key}"}
    timeout = httpx.Timeout(args.time_cap, connect=10.0)

    with httpx.Client(base_url=args.base_url, headers=headers, timeout=timeout) as client:
        def apply_experiment() -> httpx.Response | None:
            resp = client.post("/admin/api/cache-experiment", json={**experiment, "label": run_name})
            if resp.status_code == 422:  # older server without the label field
                resp = client.post("/admin/api/cache-experiment", json=experiment)
            if resp.status_code != 200:
                print(f"cache-experiment failed: {resp.status_code} {resp.text}", file=sys.stderr)
                return None
            return resp

        if not args.no_warmup:
            # One-time costs after a reset (buffer reclaim, kernel warmup) would
            # otherwise land on the first timed request. Reset again afterwards
            # so the warmup's own cache entry does not affect the run.
            if apply_experiment() is None:
                return 1
            warm = [{"role": "user", "content": "Warmup. " + src.take(512)}]
            stream_chat(client, args.model, Req("warmup", "warmup", warm), args.max_tokens)
        r = apply_experiment()
        if r is None:
            return 1
        (out_dir / "experiment.json").write_text(json.dumps(r.json(), indent=1))

        rows = []
        t_run = time.perf_counter()
        aborted = False
        with (out_dir / "requests.jsonl").open("w") as f:
            for i, req in enumerate(requests):
                if time.perf_counter() - t_run > args.time_cap:
                    aborted = True
                    break
                res = stream_chat(client, args.model, req, args.max_tokens)
                row = {"i": i, "conv": req.conv, "kind": req.kind, **res}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"{i:3d} {req.conv:>6} {req.kind:<10} prompt={res['prompt_tokens']:>6} "
                    f"cached={res['cached_tokens']:>6} prefill={res['prefilled_tokens']:>6} "
                    f"ttft={res['ttft_s']:6.2f}s",
                    flush=True,
                )
        wall = time.perf_counter() - t_run
        # Let the last async store and background SSD writes land.
        time.sleep(3.0)
        bd = client.get("/admin/api/cache-breakdown", params={"top_n": 5})
        breakdown = bd.json() if bd.status_code == 200 else {"error": bd.text}
        (out_dir / "breakdown_after.json").write_text(json.dumps(breakdown, indent=1))
        ex = client.get("/admin/api/cache-experiment")
        experiment_after = ex.json() if ex.status_code == 200 else {"error": ex.text}
        experiment_after_flush = None
        if args.final_flush:
            client.post("/admin/api/hot-cache/clear")
            time.sleep(5.0)  # background SSD writer
            exf = client.get("/admin/api/cache-experiment")
            experiment_after_flush = exf.json() if exf.status_code == 200 else {"error": exf.text}

    prompt_total = sum(r["prompt_tokens"] for r in rows)
    cached_total = sum(r["cached_tokens"] for r in rows)
    ttfts = [r["ttft_s"] for r in rows]
    model_bd = (breakdown.get("models") or {}).get(args.model) or {}
    summary = {
        "workload": args.workload,
        "label": args.label,
        "hot_cache": args.hot_cache,
        "experiment": experiment,
        "seed": args.seed,
        "requests": len(rows),
        "requests_planned": len(requests),
        "aborted_time_cap": aborted,
        "usage_missing": sum(1 for r in rows if not r["usage_present"]),
        "prompt_tokens": prompt_total,
        "cached_tokens": cached_total,
        "prefilled_tokens": prompt_total - cached_total,
        "hit_ratio": round(cached_total / prompt_total, 4) if prompt_total else 0.0,
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else 0.0,
        "ttft_p95_s": round(pct(ttfts, 0.95), 3),
        "ttft_max_s": round(max(ttfts), 3) if ttfts else 0.0,
        "wall_s": round(wall, 1),
        "probe_prefill": {
            kind: [r["prefilled_tokens"] for r in rows if r["kind"] == kind]
            for kind in sorted({r["kind"] for r in rows} - {"turn", "side"})
        },
        "gdn_snapshots_thinned": (
            (experiment_after.get("models") or {}).get(args.model, {}).get("gdn_snapshots_thinned")
        ),
        "ssd": (experiment_after.get("models") or {}).get(args.model, {}).get("ssd"),
        "gdn_sidecars_thinned": (experiment_after.get("models") or {}).get(args.model, {}).get("gdn_sidecars_thinned"),
        "staging": {
            k: (experiment_after.get("models") or {}).get(args.model, {}).get(k)
            for k in ("staging_ram_budget_bytes", "staging_ram_kept", "staging_ram_kept_bytes",
                      "staging_spilled")
        },
        "ssd_after_flush": (
            ((experiment_after_flush or {}).get("models") or {}).get(args.model, {}).get("ssd")
            if experiment_after_flush else None
        ),
        "boundary_staging_after_flush": (
            ((experiment_after_flush or {}).get("models") or {}).get(args.model, {}).get("boundary_staging_written")
            if experiment_after_flush else None
        ),
        "sidecar_boundaries_skipped": (experiment_after.get("models") or {}).get(args.model, {}).get("sidecar_boundaries_skipped"),
        "boundary_staging_written": (
            (experiment_after.get("models") or {}).get(args.model, {}).get("boundary_staging_written")
        ),
        "hot_cache_after": {
            k: model_bd.get(k)
            for k in ("hot_cache_entries", "hot_cache_bytes", "by_category",
                      "bytes_by_layer_type", "eviction_profile",
                      "eviction_policy", "admission_refusals")
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log_run(f"Finished in {summary['wall_s']} s, hit ratio {summary['hit_ratio']:.1%}")
    if not args.quiet:
        print(json.dumps({k: v for k, v in summary.items() if k != "hot_cache_after"}, indent=1))
    return 3 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
