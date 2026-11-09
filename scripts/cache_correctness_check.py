#!/usr/bin/env python3
"""Real-model correctness check for gdn_snapshot_interval (design section 3.2).

Exact equality between warm (cache-restored) and cold generations is not
guaranteed even today, because prefill chunk boundaries differ. So this
measures baseline noise first and requires thinned restores to be no worse
than today's (N=1) warm restores.

Per prompt, with greedy decoding and logprobs:
  cold_edit_a, cold_edit_b  P_edit with an empty cache, twice  -> cold/cold noise
  cold_append               P_append with an empty cache
  for N in intervals:       reset with N, send Q, then
     warm_edit_N            P_edit:   diverges from Q at its last user message,
                                      so a thinned cache must walk back to a
                                      grid block and re-prefill
     warm_append_N          P_append: extends Q, restored through Q's tail with
                                      placeholder interior blocks

Requires the server started with OMLX_CACHE_EXPERIMENTS=1.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_bench_replay import DEFAULT_CORPORA, ChunkSource  # noqa: E402

K = 1024


def build_prompts(src: ChunkSource, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    system = {"role": "system", "content": "You are a coding agent.\n" + src.take(4 * K)}
    # History sizes 8k..60k with seeded jitter so divergence points land at
    # varied offsets relative to the 2048-token block grid.
    targets = [8, 12, 16, 24, 32, 40, 50, 60][:n]
    prompts = []
    for i, t in enumerate(targets):
        hist = max(1, t - 6) * K + rng.randint(0, 1500)
        history = [
            system,
            {"role": "user", "content": f"[case {i}] Review this code:\n" + src.take(hist)},
            {"role": "assistant", "content": "Reviewed. Ready for the next file."},
        ]
        last_q = {"role": "user", "content": "Next file:\n" + src.take(rng.randint(1200, 2600))}
        q = history + [last_q]
        p_edit = history + [
            {"role": "user", "content": "Different file instead:\n" + src.take(rng.randint(600, 1400))
             + "\nExplain what this code does."}
        ]
        p_append = q + [
            {"role": "assistant", "content": "Done with that file."},
            {"role": "user", "content": "Now summarize all the code so far in one paragraph."},
        ]
        prompts.append({"id": i, "target_k": t, "q": q, "p_edit": p_edit, "p_append": p_append})
    return prompts


class Client:
    def __init__(self, base_url: str, key: str, model: str, max_tokens: int, hot_cache: str,
                 extra: dict | None = None):
        self.http = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {key}"},
                                 timeout=httpx.Timeout(600.0, connect=10.0))
        self.model = model
        self.max_tokens = max_tokens
        self.hot_cache = hot_cache
        self.extra = extra or {}

    def reset(self, interval: int) -> None:
        r = self.http.post("/admin/api/cache-experiment", json={
            "hot_cache_max_size": self.hot_cache, "reset": True,
            "gdn_snapshot_interval": interval, **self.extra,
        })
        if r.status_code != 200:
            raise SystemExit(f"cache-experiment failed: {r.status_code} {r.text}")

    def generate(self, messages: list[dict]) -> dict:
        t0 = time.perf_counter()
        r = self.http.post("/v1/chat/completions", json={
            "model": self.model, "messages": messages, "max_tokens": self.max_tokens,
            "temperature": 0.0, "logprobs": True, "top_logprobs": 1,
            "stream": False,
        })
        r.raise_for_status()
        body = r.json()
        choice = body["choices"][0]
        entries = ((choice.get("logprobs") or {}).get("content")) or []
        usage = body.get("usage") or {}
        return {
            "tokens": [e["token"] for e in entries],
            "logprobs": [e["logprob"] for e in entries],
            "text": (choice.get("message") or {}).get("content") or "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "seconds": round(time.perf_counter() - t0, 2),
        }


def compare(a: dict, b: dict) -> dict:
    """Identical-token prefix length and logprob deltas over that prefix."""
    n = 0
    for x, y in zip(a["tokens"], b["tokens"]):
        if x != y:
            break
        n += 1
    deltas = [abs(x - y) for x, y in zip(a["logprobs"][:n], b["logprobs"][:n])]
    return {
        "same_prefix": n,
        "of": min(len(a["tokens"]), len(b["tokens"])),
        "max_dlogprob": round(max(deltas), 5) if deltas else 0.0,
        "mean_dlogprob": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--hot-cache", default="8GB", help="must hold the largest Q fully")
    ap.add_argument("--intervals", default="1,2,3,4,8")
    ap.add_argument("--prompts", default="0-7", help="index range, e.g. 0-3")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra cache-experiment field applied on every reset")
    ap.add_argument("--run-log", default=None, help="append start/finish markers here")
    args = ap.parse_args()
    extra = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            extra[k] = json.loads(v)
        except json.JSONDecodeError:
            extra[k] = v

    def mark(event: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {event}: correctness check prompts {args.prompts} {extra}"
        print(line, flush=True)
        if args.run_log:
            with open(args.run_log, "a") as f:
                f.write(line + "\n")

    mark("Starting run")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    src = ChunkSource(tok, DEFAULT_CORPORA, args.seed)
    lo, _, hi = args.prompts.partition("-")
    prompts = build_prompts(src, 8, args.seed)[int(lo): int(hi or lo) + 1]
    intervals = [int(x) for x in args.intervals.split(",")]
    key = Path(args.key_file).expanduser().read_text().strip()
    client = Client(args.base_url, key, args.model, args.max_tokens, args.hot_cache, extra)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p in prompts:
        rec: dict = {"id": p["id"], "target_k": p["target_k"], "runs": {}}
        for name, msgs in (("cold_edit_a", p["p_edit"]), ("cold_edit_b", p["p_edit"]),
                           ("cold_append", p["p_append"])):
            client.reset(1)
            rec["runs"][name] = client.generate(msgs)
        for n in intervals:
            client.reset(n)
            rec["runs"][f"q_{n}"] = client.generate(p["q"])
            rec["runs"][f"warm_edit_{n}"] = client.generate(p["p_edit"])
            rec["runs"][f"warm_append_{n}"] = client.generate(p["p_append"])
        runs = rec["runs"]
        rec["noise"] = compare(runs["cold_edit_a"], runs["cold_edit_b"])
        rec["edit"] = {n: {**compare(runs[f"warm_edit_{n}"], runs["cold_edit_a"]),
                           "cached": runs[f"warm_edit_{n}"]["cached_tokens"],
                           "prompt": runs[f"warm_edit_{n}"]["prompt_tokens"]} for n in intervals}
        rec["append"] = {n: {**compare(runs[f"warm_append_{n}"], runs["cold_append"]),
                             "cached": runs[f"warm_append_{n}"]["cached_tokens"],
                             "prompt": runs[f"warm_append_{n}"]["prompt_tokens"]} for n in intervals}
        results.append(rec)
        (out_dir / f"prompt_{p['id']}.json").write_text(json.dumps(rec, indent=1))
        line = " ".join(
            f"N{n}:e{rec['edit'][n]['same_prefix']}/a{rec['append'][n]['same_prefix']}"
            f"(c{rec['edit'][n]['cached']})" for n in intervals
        )
        print(f"prompt {p['id']} ({p['target_k']}k) noise={rec['noise']['same_prefix']}/"
              f"{rec['noise']['of']} {line}", flush=True)

    # Verdict per design 3.2: warm_N no worse than warm_1 on >= all-but-one
    # prompts, and max logprob delta within 1.5x of warm_1's (floored by noise).
    verdict = {}
    for kind in ("edit", "append"):
        for n in intervals:
            if n == 1:
                continue
            worse = sum(1 for r in results if r[kind][n]["same_prefix"] < r[kind][1]["same_prefix"])
            base_max = max([r[kind][1]["max_dlogprob"] for r in results]
                           + [r["noise"]["max_dlogprob"] for r in results] + [1e-6])
            n_max = max(r[kind][n]["max_dlogprob"] for r in results)
            verdict[f"{kind}_N{n}"] = {
                "prompts_worse_than_N1": worse,
                "max_dlogprob": n_max,
                "baseline_max_dlogprob": base_max,
                "pass": worse <= max(1, len(results) // 8) and n_max <= 1.5 * base_max,
            }
    # Direct comparison: every thinned warm generation vs the N=1 warm one.
    identical = {"edit": 0, "append": 0, "different": []}
    for r in results:
        runs = r["runs"]
        for kind in ("edit", "append"):
            base = runs[f"warm_{kind}_1"]
            for n in intervals:
                if n == 1:
                    continue
                other = runs[f"warm_{kind}_{n}"]
                if other["tokens"] == base["tokens"] and other["logprobs"] == base["logprobs"]:
                    identical[kind] += 1
                else:
                    identical["different"].append((r["id"], kind, n))
    verdict["bit_identical_to_N1"] = identical
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    mark("Finished")
    return 0 if all(v["pass"] for k, v in verdict.items() if k != "bit_identical_to_N1") else 1


if __name__ == "__main__":
    sys.exit(main())
