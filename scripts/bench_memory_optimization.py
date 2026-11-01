#!/usr/bin/env python3
"""bench_memory_optimization.py — reproducible benchmark for the memory optimization pass.

Three measurements, in order of cost:

  1. DETERMINISTIC — proposal count, per-detector breakdown, wall clock over a
     real store. No model, no tokens. This is the number that has to stay small:
     a pass that emits 400 proposals per run is noise nobody reads.
  2. PROMPT SHAPE — how many targets and characters the model step would send.
     A model call costs roughly (prompt chars / 4) tokens, so this bounds the
     per-run price before any request is made.
  3. REPLY VALIDATION — parse a model reply (--reply FILE) through the same
     validator the production path uses, and report what survived: proposals
     accepted, additions that could be traced back to the file text, targets the
     validator refused. A model that invents keywords never gets them applied;
     this measures how often that happens instead of trusting the prompt.

Usage:
    python3 scripts/bench_memory_optimization.py                       # 1 + 2
    python3 scripts/bench_memory_optimization.py --reply out.json      # also 3
    python3 scripts/bench_memory_optimization.py --write-prompt p.txt  # emit the prompt
    python3 scripts/bench_memory_optimization.py --store DIR --db PATH --session ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.memory_optimization import (  # noqa: E402
    MAX_KEYWORDS,
    MIN_KEYWORDS,
    analyze,
    build_refine_request,
    load_index,
    parse_refine_reply,
    report,
    summary_line,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(Path.home() / ".hermes" / "memories"))
    ap.add_argument("--db", default=str(Path.home() / ".hermes" / "state.db"))
    ap.add_argument("--session", help="include the session-drift detector")
    ap.add_argument("--reply", help="model reply file to validate")
    ap.add_argument("--write-prompt", help="write the model prompt here")
    ap.add_argument("--limit", type=int, default=4, help="targets sent to the model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    store = Path(args.store).expanduser()
    db = Path(args.db).expanduser() if args.db else None
    if not (store / "MEMORY.md").exists():
        print(f"ERROR: no index at {store / 'MEMORY.md'}", file=sys.stderr)
        return 2

    lines, entries = load_index(store)
    messages = None
    scope = None
    if args.session:
        from agent.memory_optimization import _load_session_messages, session_touched_files
        messages = _load_session_messages(db, args.session)
        scope = list(session_touched_files(messages, store).keys())

    t0 = time.perf_counter()
    props = analyze(store, messages=messages, scope=scope, db=db)
    elapsed = time.perf_counter() - t0

    kinds = Counter(p.kind for p in props)
    auto = sum(1 for p in props if p.auto_applicable)
    result = {
        "store": str(store),
        "entries": len(entries),
        "proposals": len(props),
        "by_kind": dict(kinds),
        "auto_applicable": auto,
        "seconds": round(elapsed, 3),
        "proposals_per_second": round(len(props) / elapsed, 1) if elapsed else None,
    }

    prompt, targets = build_refine_request(store, props, limit=args.limit)
    result["model_targets"] = targets
    result["prompt_chars"] = len(prompt)
    result["prompt_tokens_est"] = len(prompt) // 4
    if args.write_prompt:
        Path(args.write_prompt).write_text(prompt)

    if args.reply:
        reply = Path(args.reply).read_text()
        accepted = parse_refine_reply(reply, targets, store)
        # an addition that cannot be found in the file body is dropped by the
        # validator; count surfaced vs accepted to show the filter working
        result["reply_chars"] = len(reply)
        result["reply_accepted"] = len(accepted)
        result["reply_detail"] = [
            {"target": p.target, "add": p.add, "remove": p.remove, "why": p.detail}
            for p in accepted
        ]
        bad = [
            p for p in accepted
            if not (MIN_KEYWORDS <= len([k for k in _kw_of(store, p.target) if k not in p.remove] + p.add) <= MAX_KEYWORDS)
        ]
        result["keyword_bounds_violations"] = len(bad)

    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"store      {store}")
        print(f"entries    {len(entries)}")
        print(f"proposals  {len(props)}  ({auto} auto-applicable, rest review-only)")
        print(f"by kind    {dict(kinds)}")
        print(f"speed      {len(props)} proposals in {elapsed:.2f}s "
              f"({result['proposals_per_second']}/s)")
        print(f"model step {len(targets)} target(s), {len(prompt)} chars "
              f"(~{len(prompt)//4} tokens): {', '.join(targets) or '-'}")
        if args.reply:
            print(f"reply      {result['reply_chars']} chars -> "
                  f"{result['reply_accepted']} proposal(s) accepted, "
                  f"{result['keyword_bounds_violations']} bound violation(s)")
            for d in result["reply_detail"]:
                print(f"  {d['target']:34s} +{d['add']} -{d['remove']}  ({d['why']})")
        print("\n" + report(props).split("\n", 1)[-1][:1200])
        print("\n" + summary_line(props))
    return 0


def _kw_of(store: Path, target: str) -> list[str]:
    _, entries = load_index(store)
    for e in entries:
        if e.target == target:
            return e.keywords
    return []


if __name__ == "__main__":
    raise SystemExit(main())
