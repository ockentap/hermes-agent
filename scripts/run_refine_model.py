#!/usr/bin/env python3
"""run_refine_model.py — run the model step from THIS repo, no Hermes config involved.

Why this exists: the first benchmark drove the model through the running agent's
subagent path, which meant temporarily editing the production delegation config.
That is the wrong place to run an experiment. This script talks to an
OpenAI-compatible endpoint directly, takes the prompt from a file, validates the
reply with the production validator, and prints metrics. Nothing outside this
repo (and the store you point it at) is touched.

Usage:
    export OPENCODE_API_KEY=...            # or pass --api-key
    python3 scripts/run_refine_model.py \
        --prompt /tmp/refine_prompt.txt \
        --base-url https://opencode.ai/zen/go/v1 \
        --model deepseek-v4.1-flash \
        --store ~/.hermes/memories \
        --out /tmp/reply.json

Exit codes: 0 ok, 2 unusable reply, 3 missing credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.memory_optimization import (  # noqa: E402
    analyze,
    build_refine_request,
    parse_refine_reply,
)


def call_model(base_url: str, api_key: str, model: str, prompt: str,
               timeout: int = 120, temperature: float = 0.0) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = payload.get("usage") or {}
    return {"text": text, "seconds": round(elapsed, 2), "usage": usage}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="prompt file (see bench --write-prompt)")
    ap.add_argument("--base-url", default=os.environ.get("REFINE_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("REFINE_MODEL", "deepseek-v4.1-flash"))
    ap.add_argument("--api-key", default=os.environ.get("OPENCODE_API_KEY")
                    or os.environ.get("REFINE_API_KEY", ""))
    ap.add_argument("--store", default=str(Path.home() / ".hermes" / "memories"))
    ap.add_argument("--out", help="write the raw reply here")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)

    if not args.base_url:
        print("ERROR: --base-url (or REFINE_BASE_URL) is required", file=sys.stderr)
        return 3
    if not args.api_key:
        print("ERROR: no key. Set OPENCODE_API_KEY / REFINE_API_KEY or pass --api-key.\n"
              "       This script deliberately does not read the agent's credential store:\n"
              "       a benchmark must not depend on production secrets or production config.",
              file=sys.stderr)
        return 3

    prompt = Path(args.prompt).read_text()
    try:
        res = call_model(args.base_url, args.api_key, args.model, prompt, timeout=args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read()[:400]!r}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"network: {exc}", file=sys.stderr)
        return 2

    text = res["text"]
    if args.out:
        Path(args.out).write_text(text)

    store = Path(args.store).expanduser()
    props = analyze(store)
    _, targets = build_refine_request(store, props, limit=4)
    try:
        raw = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        print(f"model={args.model} seconds={res['seconds']} "
              f"usage={res['usage']} -> NOT VALID JSON", file=sys.stderr)
        return 2
    accepted = parse_refine_reply(text, targets, store)
    proposed_add = sum(len(i.get("add") or []) for i in raw.get("items", []))
    proposed_rm = sum(len(i.get("remove") or []) for i in raw.get("items", []))
    print(f"model={args.model} seconds={res['seconds']} usage={res['usage']}")
    print(f"proposed +{proposed_add} -{proposed_rm} | accepted "
          f"+{sum(len(p.add) for p in accepted)} -{sum(len(p.remove) for p in accepted)}")
    for p in accepted:
        print(f"  {p.target:34s} +{p.add} -{p.remove}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
