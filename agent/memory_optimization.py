"""Memory optimization pass — session-scoped keyword hygiene for the keyword-index memory.

Companion to :mod:`agent.background_review` (which asks a forked agent "should any
skill or memory be saved?"). This module answers a narrower, cheaper question:

    "the memories this session created or touched — are their keywords still right,
     now that the session has moved on?"

It never restructures a topic file and never reorders the index (that is the
access watchdog's job). It emits proposals. Applying them means editing the
keyword list of one index line, which the caller does through the memory tool so
the write lock and the index format validator still see it.

Design constraints, in order:
  1. Stdlib only, no imports from the rest of Hermes — the analyzer must be
     runnable standalone against any store, which is also what makes it
     benchmarkable without booting an agent.
  2. Deterministic first. Eight of the nine detectors need no model at all, cost
     nothing, and cannot hallucinate. The model-assisted path is opt-in and only
     decorates proposals it is given; it never invents a target file.
  3. Never propose a write to a tagged line (``!``, ``!!``, ``!!!``) — those are
     user-owned. They are reported, not changed.

Store layout expected (see the dynamic-memory architecture):

    <store>/MEMORY.md                  keyword index, one line per topic
    <store>/<topic>.md                 topic files
    <store>/.access-tally.json         optional: reads/day buckets (watchdog)

Index line grammar: ``keyword,keyword,keyword -> topic.md`` with an optional
``!``/``!!``/``!!!`` qualifier immediately before the keyword list.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

ARROW = "\u2192"
INDEX_NAME = "MEMORY.md"
MIN_KEYWORDS = 5
MAX_KEYWORDS = 20
BODY_READ_LIMIT = 200_000

# Tokens that make a keyword "specific": an identifier, a number, a host, a
# version, a product code. These are the ones a model cannot infer on its own, so
# they are the highest-value things to have on a line.
_SPECIFIC_RE = re.compile(r"""[A-Za-z0-9][A-Za-z0-9._/:-]*""")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "then",
    "than", "they", "them", "there", "here", "what", "which", "will", "would",
    "should", "could", "have", "has", "had", "was", "were", "are", "not", "but",
    "you", "your", "our", "its", "it's", "all", "any", "can", "may", "one", "two",
    "use", "used", "using", "make", "made", "get", "got", "set", "run", "ran",
    "file", "files", "test", "tests", "true", "false", "none", "null", "return",
}


@dataclass
class Entry:
    pos: int                 # 0-based line index in the index file
    raw: str                 # the line, verbatim
    marker: str              # "", "!", "!!", "!!!"
    keywords: list[str]
    target: str              # "topic.md"

    @property
    def tagged(self) -> bool:
        return bool(self.marker)


@dataclass
class Proposal:
    kind: str                # detector id
    target: str              # topic file the proposal is about
    detail: str              # one human sentence
    evidence: str = ""       # why we think so (quoted token / line fragment)
    add: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    confidence: float = 0.5
    auto_applicable: bool = False   # False for tagged lines or low confidence
    writes_body: bool = False       # True only for the optional model pass

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- store --------

def parse_line(line: str) -> Optional[Entry]:
    s = line.strip()
    if ARROW not in s:
        return None
    left, right = s.split(ARROW, 1)
    marker = ""
    for m in ("!!!", "!!", "!"):
        if left.startswith(m):
            marker = m
            left = left[len(m):]
            break
    keywords = [k.strip() for k in left.split(",") if k.strip()]
    target = right.strip()
    if not target.endswith(".md"):
        return None
    # The index ships with a literal template line ("keyword,keyword,keyword -> file.md").
    # It is documentation, not a routing entry.
    if target == "file.md" or (len(keywords) > 1 and all(k == "keyword" for k in keywords)):
        return None
    return Entry(pos=-1, raw=line, marker=marker, keywords=keywords, target=target)


def load_index(store: Path) -> tuple[list[str], list[Entry]]:
    """Return (raw_lines, entries). Non-entry lines are kept for context (header)."""
    path = store / INDEX_NAME
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries: list[Entry] = []
    for i, line in enumerate(lines):
        e = parse_line(line)
        if e is not None:
            e.pos = i
            entries.append(e)
    return lines, entries


def load_body(store: Path, target: str) -> str:
    p = store / target
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(BODY_READ_LIMIT)
    except OSError:
        return ""


def header_text(lines: Sequence[str], entries: Sequence[Entry]) -> str:
    """Everything in the index that is not an entry line."""
    taken = {e.pos for e in entries}
    return "\n".join(l for i, l in enumerate(lines) if i not in taken)


def keyword_owners(entries: Sequence[Entry]) -> dict[str, list[str]]:
    own: dict[str, list[str]] = {}
    for e in entries:
        for k in e["keywords"] if isinstance(e, dict) else e.keywords:
            own.setdefault(k, []).append(e["target"] if isinstance(e, dict) else e.target)
    return own


# ------------------------------------------------------------ detectors -------

_BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_PATH_RE = re.compile(r"\b(?:[\w.-]+/){1,6}[\w.-]+\b")
_SNAKE_RE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")
_DOTTED_RE = re.compile(r"\b[a-z0-9]+(?:\.[a-z0-9]+){2,}\b")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_PLACEHOLDER_RE = re.compile(r"^(?:path/to/|/path/to/|foo|bar|baz|example|xxx|your[-_]|<|\$\{|dev/null|dev/sd[xn]|python3?$|node$|bash$|true$|false$)")
_PARTIAL_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){0,2}$")
# Random ids, flags, units and CLI fragments are never routing keywords.
_JUNK_RE = re.compile(
    r"^(?:--?[a-z-]+|mb/s|kb/s|gb/s|-|\+|[a-z]{2,6}_[0-9a-f]{16,}|"
    r"[a-z]*(?:skill|plugin)s?/[a-z0-9-]+|complete|done|started|failure)$"
)


def _identifier_tokens(body: str) -> list[str]:
    """Tokens a model cannot infer from the topic: code spans, paths, IPs, versions.

    Prose words are deliberately excluded. A 7-letter English word is not a
    keyword — it is noise that would make every line a synonym soup. The only
    additions worth proposing are the strings that identify *this* thing.
    """
    low = body.lower()
    out: list[str] = []
    for m in _BACKTICK_RE.finditer(low):
        chunk = m.group(1).strip()
        if 2 <= len(chunk) <= 60 and " " not in chunk:
            out.append(chunk)
        else:
            out.extend(t for t in _SPECIFIC_RE.findall(chunk) if len(t) >= 4)
    for rx in (_IP_RE, _PATH_RE, _SNAKE_RE, _DOTTED_RE, _VERSION_RE):
        out.extend(rx.findall(low))
    clean = []
    for t in out:
        t = t.strip("`'\"(),.;:[]{}")
        if len(t) < 3 or t in _STOPWORDS:
            continue
        if _PLACEHOLDER_RE.match(t) or _PARTIAL_IP_RE.match(t) or _JUNK_RE.match(t):
            continue
        if t.count("/") >= 3 or len(t) > 42:
            continue
        if not any(c.isdigit() for c in t) and not any(c in t for c in "._/:") and "_" not in t:
            # a bare word is only kept when it came from a code span AND is long
            # enough to be an identifier rather than prose
            if len(t) < 8:
                continue
        clean.append(t)
    return clean


def _fold(text: str) -> str:
    """ASCII-fold accents so 'omerovic' matches 'Omerović' and 'kallor' matches 'källor'."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    ).lower()


def _appears(keyword: str, body_low: str) -> bool:
    """Loose containment: 'backups' matches 'backup', 'vps' matches 'vps' only.

    Deliberately conservative — a prefix match catches plural/gerund drift without
    a stemmer, and refusing to match is the safe direction (we then say nothing).
    """
    k = _fold(keyword).strip()
    if not k:
        return True
    body = _fold(body_low)
    if k in body:
        return True
    stem = k.rstrip("s")
    if len(stem) >= 4 and stem in body:
        return True
    # an alias may be a different surface form of a word in the body
    if len(k) >= 5 and k[:5] in body:
        return True
    return False


_CORPUS_SPLIT = re.compile(r"[^a-z0-9_.:/-]+")


def user_corpus_tokens(db: Optional[Path], days: int = 90) -> set[str]:
    """Folded token set from the user's own messages in the window.

    A set, not a string: membership has to be O(1). The first version scanned a
    20 MB concatenation once per keyword and took five minutes on a 133-entry store.
    """
    if not db or not Path(db).exists():
        return set()
    import sqlite3
    import time as _time
    since = _time.time() - days * 86400
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=20)
        rows = con.execute(
            "SELECT content FROM messages WHERE role = 'user' AND timestamp >= ? "
            "AND content IS NOT NULL", (since,)
        ).fetchall()
        con.close()
    except Exception:
        return set()
    toks: set[str] = set()
    for (c,) in rows:
        if not c:
            continue
        toks.update(t for t in _CORPUS_SPLIT.split(_fold(c)) if t)
    return toks


def detect_coverage(store: Path, entries: Sequence[Entry], corpus: Optional[set[str]] = None) -> list[Proposal]:
    """Keywords with no trace in the file, and specific body terms missing from the line."""
    out: list[Proposal] = []
    owners = keyword_owners(entries)
    for e in entries:
        body = load_body(store, e.target)
        if not body:
            continue
        low = body.lower()
        # Unused = absent from the file body AND absent from every user message in
        # the window. Absent-from-body alone is meaningless: 'cv', 'cli', 'zlato'
        # and 'dug' are exactly the words the user types and the file never says.
        dead = []
        if corpus:
            for k in e.keywords:
                if _appears(k, low) or len(owners.get(k, [])) > 1:
                    continue
                fk = _fold(k).strip()
                if fk and fk not in corpus and fk.rstrip("s") not in corpus:
                    dead.append(k)
        if dead and len(dead) < len(e.keywords):
            out.append(Proposal(
                kind="coverage-dead-keyword",
                target=e.target,
                detail=f"{len(dead)} keyword(s) never appear in the file body",
                evidence=", ".join(dead[:8]),
                remove=dead[:8],
                confidence=0.6 if len(dead) <= 3 else 0.35,
                auto_applicable=not e.tagged,
            ))
        # Missing specific terms: identifiers/numbers used repeatedly in the body
        # that no index line carries at all.
        counts: dict[str, int] = {}
        for t in _identifier_tokens(body):
            counts[t] = counts.get(t, 0) + 1
        missing = [
            t for t, n in sorted(counts.items(), key=lambda kv: -kv[1])
            if n >= 2
            and t not in owners                       # on no index line at all
            and not any(t in k or k in t for k in e.keywords)
            and len(t) <= 48
        ][:4]
        if missing:
            out.append(Proposal(
                kind="coverage-missing-keyword",
                target=e.target,
                detail=f"{len(missing)} specific term(s) in the body are on no index line",
                evidence=", ".join(f"{t}(x{counts[t]})" for t in missing[:5]),
                add=missing,
                confidence=0.55,
                auto_applicable=not e.tagged and len(e.keywords) < MAX_KEYWORDS,
            ))
    return out


def detect_orphans(store: Path, entries: Sequence[Entry], lines: Sequence[str]) -> list[Proposal]:
    """Topic files with no routing line, and lines whose file is gone."""
    out: list[Proposal] = []
    targets = {e.target for e in entries}
    header = header_text(lines, entries).lower()
    for f in sorted(store.glob("*.md")):
        if f.name == INDEX_NAME or f.name.startswith("USER"):
            continue
        if f.name in targets:
            continue
        if f.name.lower() in header:            # reachable from the header prose
            continue
        size = f.stat().st_size
        out.append(Proposal(
            kind="orphan-file",
            target=f.name,
            detail=f"topic file has no index line and is not referenced in the header ({size} B)",
            evidence=f.name,
            confidence=0.9,
            auto_applicable=True,
        ))
    for e in entries:
        if not (store / e.target).exists():
            out.append(Proposal(
                kind="orphan-line",
                target=e.target,
                detail="index line points at a file that does not exist",
                evidence=e.raw[:120],
                confidence=0.95,
                auto_applicable=not e.tagged,
            ))
    seen: dict[str, int] = {}
    for e in entries:
        seen[e.target] = seen.get(e.target, 0) + 1
    for tgt, n in seen.items():
        if n > 1:
            out.append(Proposal(
                kind="duplicate-target",
                target=tgt,
                detail=f"{n} index lines point at the same file",
                evidence=tgt,
                confidence=0.8,
                auto_applicable=False,
            ))
    return out


def detect_duplicate_keywords(entries: Sequence[Entry], scope: Optional[Iterable[str]] = None) -> list[Proposal]:
    """Same keyword on more than one line. Scoped to `scope` targets when given."""
    out: list[Proposal] = []
    own = keyword_owners(entries)
    keep = set(scope) if scope is not None else None
    for kw, tgts in own.items():
        if len(tgts) < 2:
            continue
        if keep is not None and not (set(tgts) & keep):
            continue
        out.append(Proposal(
            kind="duplicate-keyword",
            target=", ".join(sorted(tgts)[:4]),
            detail=f"keyword '{kw}' routes to {len(tgts)} files",
            evidence=kw,
            remove=[kw],
            confidence=0.4,
            auto_applicable=False,
        ))
    return out


# ------------------------------------------------------ distinctive terms ----

_WORD_RE = re.compile(r"[a-z][a-z0-9_./-]{3,}")
# Words too common in English/tech prose to route anything on their own. This is
# deliberately small: the df==1 test below is the real filter.
_FOREIGN_STOPWORDS = {
    # Swedish (Lav's documents), Bosnian/BCS, German, Spanish — housekeeping words
    # that would otherwise pass as "unique" in a file written in that language.
    "ingen", "inte", "inga", "och", "att", "som", "for", "med", "det", "den", "ar",
    "till", "har", "kan", "om", "vi", "jag", "fran", "efter", "eller", "enligt",
    "je", "da", "se", "na", "za", "od", "sa", "koji", "nije", "ali", "ili", "kao",
    "kod", "pri", "kada", "ako", "samo", "biti", "ima", "nema", "treba", "mora",
    "der", "die", "das", "und", "nicht", "eine", "einen", "auch", "wird", "sind",
    "para", "como", "pero", "cuando", "desde", "sobre", "todo", "esta",
}
_GENERIC_WORDS = {
    "this", "that", "with", "from", "have", "will", "when", "then", "than", "they",
    "them", "there", "their", "which", "while", "what", "were", "your", "should",
    "could", "would", "about", "after", "before", "because", "other", "into",
    "file", "files", "line", "lines", "note", "notes", "true", "false", "none",
    "work", "works", "working", "used", "using", "make", "made", "need", "needs",
    "same", "only", "also", "very", "much", "more", "most", "some", "such",
    "here", "does", "done", "each", "both", "must", "over", "under", "again",
    "check", "checked", "value", "values", "thing", "things", "case", "cases",
    "step", "steps", "first", "second", "last", "next", "back", "down", "up",
}


def _doc_vocab(body: str) -> dict[str, int]:
    """word -> count, folded, stopwords and generic prose removed."""
    counts: dict[str, int] = {}
    for raw in _WORD_RE.findall(_fold(body)):
        raw = raw.strip("._/:-")
        if (raw in _STOPWORDS or raw in _GENERIC_WORDS
                or raw in _FOREIGN_STOPWORDS or len(raw) < 5):
            continue
        if raw.isdigit():
            continue
        counts[raw] = counts.get(raw, 0) + 1
    return counts


MAX_AUTO_KEYWORDS = 5          # hard cap per file: auto-mined keywords
PROVENANCE_NAME = ".keyword-provenance.json"


def provenance_path(store: Path) -> Path:
    return store / PROVENANCE_NAME


def load_provenance(store: Path) -> dict:
    """{file.md: {keyword: {"source": ..., "added": ..., "score_at_add": ...}}}"""
    try:
        raw = json.loads(provenance_path(store).read_text())
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def save_provenance(store: Path, data: dict) -> None:
    tmp = provenance_path(store).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(provenance_path(store))


def record_applied(store: Path, file_target: str, added: Sequence[str],
                   removed: Sequence[str] = (), now: Optional[str] = None) -> None:
    """Update the provenance sidecar after a keyword edit is actually applied.

    Only this sidecar knows which keywords were machine-mined; without it the cap
    and the swap policy have nothing to count, and a hand-written keyword would be
    evicted in favour of a generated one.
    """
    data = load_provenance(store)
    ent = data.setdefault(file_target, {})
    for k in removed:
        ent.pop(k, None)
    stamped = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for k in added:
        ent[k] = {"source": "unique-term", "added": stamped}
    if not ent:
        data.pop(file_target, None)
    save_provenance(store, data)


def _term_score(word: str, counts: dict, body: str) -> float:
    """Frequency, plus a bump when the term appears in the file's title/lede."""
    score = float(counts.get(word, 0))
    if word in _fold(body[:200]):
        score += 5.0
    return score


def detect_unique_terms(store: Path, entries: Sequence[Entry],
                        limit_per_file: int = MAX_AUTO_KEYWORDS) -> list[Proposal]:
    """Words that identify exactly one memory file, capped at five per file.

    Cap and replacement policy:
      * at most MAX_AUTO_KEYWORDS machine-mined keywords per file at any time;
      * the count comes from the provenance sidecar, so hand-written keywords are
        never evicted to make room;
      * when the file is at the cap, a stronger candidate displaces the weakest
        tracked one — a swap (add + remove), never a sixth addition.

    A word already on any index line is skipped, so "router" on the openwrt line is
    never proposed again. Document frequency of 1 is the distinctiveness test;
    no dictionary or model is involved.
    """
    cap = max(0, min(limit_per_file, MAX_AUTO_KEYWORDS))
    owners = keyword_owners(entries)
    bodies = {e.target: load_body(store, e.target) for e in entries}
    vocab = {t: _doc_vocab(b) for t, b in bodies.items() if b}
    df: dict[str, int] = {}
    for counts in vocab.values():
        for word in counts:
            df[word] = df.get(word, 0) + 1
    tracked = load_provenance(store)
    out: list[Proposal] = []

    for e in entries:
        counts = vocab.get(e.target)
        if not counts:
            continue
        body = bodies[e.target]
        mine = tracked.get(e.target, {})
        # tracked keywords still present on the line count against the cap
        active = {k: v for k, v in mine.items() if k in e.keywords}
        cands = [
            (word, _term_score(word, counts, body)) for word in counts
            if df.get(word) == 1
            and counts[word] >= 3
            and word not in owners
            and word not in e.keywords
            and len(word) >= 5
        ]
        if not cands:
            continue
        cands.sort(key=lambda ws: -ws[1])
        room = cap - len(active)
        if room > 0:
            picks = [w for w, _ in cands[:room]]
            out.append(Proposal(
                kind="unique-term",
                target=e.target,
                detail=(f"{len(picks)} unique word(s) added "
                        f"({len(active)}/{cap} auto-keyword slots used)"),
                evidence=", ".join(f"{w}({counts[w]})" for w in picks),
                add=picks,
                confidence=0.6,
                auto_applicable=not e.tagged and len(e.keywords) + len(picks) <= MAX_KEYWORDS,
            ))
            continue
        # at the cap: swap only if the best candidate beats the weakest tracked word
        if not active or e.tagged:
            continue
        weakest = min(active, key=lambda k: (_term_score(k, counts, body),
                                             active[k].get("added", "")[:10]))
        best_word, best_score = cands[0]
        weakest_score = _term_score(weakest, counts, body)
        # Require a real margin, not a one-mention lead: swapping is churn, and a
        # keyword that was good enough yesterday should not lose to noise.
        if best_score < weakest_score * 1.5 + 1:
            continue
        out.append(Proposal(
            kind="unique-term-swap",
            target=e.target,
            detail=(f"auto-keyword cap reached ({cap}/{cap}); "
                    f"'{best_word}' ({counts.get(best_word, 0)}) outranks '{weakest}' "
                    f"({counts.get(weakest, 0)})"),
            evidence=f"+{best_word} -{weakest}",
            add=[best_word],
            remove=[weakest],
            confidence=0.55,
            auto_applicable=True,
        ))
    return out


def detect_thin_lines(entries: Sequence[Entry]) -> list[Proposal]:
    """Lines outside the 5..20 keyword band the index format requires."""
    out = []
    for e in entries:
        n = len(e.keywords)
        if n < MIN_KEYWORDS or n > MAX_KEYWORDS:
            out.append(Proposal(
                kind="keyword-count",
                target=e.target,
                detail=f"{n} keywords (index format expects {MIN_KEYWORDS}-{MAX_KEYWORDS})",
                evidence=e.raw[:120],
                confidence=0.9,
                auto_applicable=False,
            ))
    return out


# --------------------------------------------------------- session drift ------

def session_touched_files(messages: Sequence[dict], store: Path) -> dict[str, int]:
    """Map topic file -> index of the last message that mentioned it in a tool call."""
    names = {p.name for p in store.glob("*.md")}
    touched: dict[str, int] = {}
    for i, m in enumerate(messages):
        blob = json.dumps(m.get("tool_calls") or "") + " " + str(m.get("content") or "")
        for name in names:
            if name in blob:
                touched[name] = i
    return touched


def detect_session_drift(
    store: Path,
    entries: Sequence[Entry],
    messages: Sequence[dict],
) -> list[Proposal]:
    """Vocabulary the session used *after* touching a file that its line does not carry.

    This is the case the whole pass exists for: a memory written twenty minutes ago,
    then the conversation moved slightly and the line no longer describes it.
    """
    out: list[Proposal] = []
    owners = keyword_owners(entries)
    by_target = {e.target: e for e in entries}
    touched = session_touched_files(messages, store)
    for target, last_idx in touched.items():
        e = by_target.get(target)
        if e is None:
            continue
        later = " ".join(
            str(m.get("content") or "") for m in messages[last_idx + 1:]
            if m.get("role") == "user"
        )
        if len(later) < 40:
            continue
        low = later.lower()
        cand_counts: dict[str, int] = {}
        for t in _identifier_tokens(later):
            if any(t in k or k in t for k in e.keywords):
                continue
            if t in owners:                       # belongs to another line
                continue
            cand_counts[t] = cand_counts.get(t, 0) + 1
        picks = [
            t for t, n in sorted(cand_counts.items(), key=lambda kv: -kv[1])
            if n >= 2 and (any(c.isdigit() for c in t) or "_" in t or "." in t
                           or "/" in t or len(t) >= 6)
        ][:4]
        if picks:
            out.append(Proposal(
                kind="session-drift",
                target=target,
                detail=(
                    f"used later in the session but absent from the line "
                    f"({len(picks)} term(s), after msg #{last_idx})"
                ),
                evidence=", ".join(picks),
                add=picks,
                confidence=0.5,
                auto_applicable=not e.tagged and len(e.keywords) < MAX_KEYWORDS,
            ))
    return out


# ------------------------------------------------------------- reporting ------

_GLYPH = "💾"


def summary_line(proposals: Sequence[Proposal], max_targets: int = 3) -> str:
    """One short chat line: "🧠 Memory optimization: 3 memories optimized".

    The detail (which keywords, which files) belongs in the report, not the chat
    line — a long line in every turn is noise the user stops reading.
    """
    if not proposals:
        return ""
    targets = {p.target for p in proposals if p.add or p.remove}
    n = len(targets)
    if not n:
        return ""
    return f"\U0001F9E0 Memory optimization: {n} memor{'y' if n == 1 else 'ies'} optimized"


def applied_note(applied: int) -> str:
    return f" · {applied} applied" if applied else ""


def report(proposals: Sequence[Proposal]) -> str:
    """Multi-line human report for --report."""
    if not proposals:
        return "no proposals"
    by_kind: dict[str, list[Proposal]] = {}
    for p in proposals:
        by_kind.setdefault(p.kind, []).append(p)
    lines = []
    for kind in sorted(by_kind, key=lambda k: (-len(by_kind[k]), k)):
        items = by_kind[kind]
        lines.append(f"\n{kind} ({len(items)})")
        for p in items[:12]:
            flag = "" if p.auto_applicable else "  [manual]"
            lines.append(f"  {p.target:38s} {p.detail}{flag}")
            if p.evidence:
                lines.append(f"      evidence: {p.evidence[:160]}")
    return "\n".join(lines)


def analyze(
    store: Path,
    messages: Optional[Sequence[dict]] = None,
    scope: Optional[Iterable[str]] = None,
    kinds: Optional[Iterable[str]] = None,
    db: Optional[Path] = None,
    only: Optional[Iterable[str]] = None,
) -> list[Proposal]:
    """Run the detectors.

    `scope` limits the session/duplicate detectors to given files.
    `only` limits the whole pass to given files — the per-turn path uses it so the
    cost is a few files instead of the entire store.
    """
    lines, entries = load_index(store)
    if only is not None:
        keep = set(only)
        entries = [e for e in entries if e.target in keep]
    corpus = user_corpus_tokens(db) if db else set()
    props: list[Proposal] = []
    props += detect_coverage(store, entries, corpus=corpus)
    props += detect_orphans(store, entries, lines)
    props += detect_unique_terms(store, entries)
    props += detect_thin_lines(entries)
    if messages:
        props += detect_session_drift(store, entries, messages)
        # collisions are only interesting for the files this session touched;
        # the global sweep is the daily job's job, not a per-turn one
        props += detect_duplicate_keywords(entries, scope=scope)
    if kinds:
        keep = set(kinds)
        props = [p for p in props if p.kind in keep]
    return props


# ----------------------------------------------------------- model pass ------
#
# The model is a *decorator*, not a generator: it receives proposals that already
# exist with their evidence and may merge, re-word, or drop them. It is never
# asked "what keywords should this file have?", because that invites invention.

REFINE_PROMPT = """You are tightening the keyword routing line of a memory file.

For each item you get: the current line, the file's opening lines, and candidate
keyword changes found by deterministic analysis. Your job:

- drop candidates that are noise (too generic to route anything, or already
  implied by an existing keyword)
- keep at most 4 additions per file, preferring identifiers, hosts, versions,
  product codes, error strings and proper nouns
- keep every line between 5 and 20 keywords
- never invent a keyword that does not appear in the file text or the quoted
  evidence

Return STRICT JSON only, no prose:
{"items":[{"target":"x.md","add":["..."],"remove":["..."],"why":"<=20 words"}]}
Omit a target entirely if nothing should change.

"""


def build_refine_request(store: Path, proposals: Sequence[Proposal], limit: int = 8) -> tuple[str, list[str]]:
    """Prompt + the list of targets it covers (so the reply can be validated)."""
    groups: dict[str, list[Proposal]] = {}
    for p in proposals:
        if p.kind in ("session-drift", "coverage-missing-keyword", "coverage-dead-keyword"):
            groups.setdefault(p.target, []).append(p)
    _, entries = load_index(store)
    by_target = {e.target: e for e in entries}
    blocks, targets = [], []
    # Tagged lines are user-owned: never send them to a model that might propose
    # an edit. Prefer untagged targets even when a tagged one sorts first.
    eligible = [t for t in groups if t in by_target and not by_target[t].tagged]
    for target in eligible[:limit]:
        items = groups[target]
        e = by_target[target]
        head = " ".join(load_body(store, target).split())[:600]
        cands = []
        for p in items:
            if p.add:
                cands.append(f"- suggested add: {', '.join(p.add)} ({p.evidence[:120]})")
            if p.remove:
                cands.append(f"- suggested remove: {', '.join(p.remove)} ({p.evidence[:120]})")
        blocks.append(
            f"### {target}\nline: {e.raw}\nfile starts: {head}\n" + "\n".join(cands)
        )
        targets.append(target)
    return REFINE_PROMPT + "\n\n".join(blocks), targets


def parse_refine_reply(
    text: str,
    allowed_targets: Sequence[str],
    store: Path,
    justified_removals: Optional[dict[str, set[str]]] = None,
) -> list[Proposal]:
    """Validate a model reply. Anything not in `allowed_targets` or the file text is dropped.

    Removals are only honoured when `justified_removals` says the deterministic pass
    already flagged that exact keyword on that exact file. Measured reason: given
    free rein, two runs independently proposed dropping `open-medivia` (the project
    name), `patch` (from the file about local patches), and `pentest`/`audit` — the
    words a user actually asks with. Additions were pure echoes of the candidates
    they were handed; removals were invention. So the model may add, and may drop a
    candidate from the add list, but it may not take words away.
    """
    out: list[Proposal] = []
    try:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return out
    _, entries = load_index(store)
    by_target = {e.target: e for e in entries}
    for item in data.get("items", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target", ""))
        if target not in allowed_targets or target not in by_target:
            continue
        e = by_target[target]
        if e.tagged:
            continue
        body_low = load_body(store, target).lower()
        add = [k for k in (item.get("add") or []) if isinstance(k, str) and k.strip()]
        # an addition must be traceable to the file itself
        add = [k for k in add if k.lower() in body_low][:4]
        justified = (justified_removals or {}).get(target, set())
        remove = [k for k in (item.get("remove") or [])
                  if k in e.keywords and k in justified]
        if not add and not remove:
            continue
        if len(e.keywords) - len(remove) + len(add) > MAX_KEYWORDS:
            headroom = max(0, MAX_KEYWORDS - (len(e.keywords) - len(remove)))
            add = add[:headroom]          # trim additions, never evict existing words
        if len(e.keywords) - len(remove) + len(add) < MIN_KEYWORDS:
            continue                    # would strip the line below the format floor
        out.append(Proposal(
            kind="llm-refined",
            target=target,
            detail=str(item.get("why", ""))[:120] or "model-refined",
            evidence=", ".join(f"+{a}" for a in add) + (", " + ", ".join(f"-{r}" for r in remove) if remove else ""),
            add=add,
            remove=remove,
            confidence=0.55,
            auto_applicable=True,
        ))
    return out


# ------------------------------------------------------------- applying ------

def apply_plan(store: Path, proposals: Sequence[Proposal]) -> list[dict]:
    """Turn keyword proposals into the exact edits a memory-tool call can make.

    MEMORY.md is write-locked for everything except the memory tool, so the applier
    cannot be a plain file write. Each entry here is one `memory(action="patch")`:
    `old_text` is the line's whole keyword span (unique inside the file), `new_text`
    is the replacement span. The caller applies them and then calls
    `record_applied()` so the provenance sidecar knows what was machine-mined.
    """
    _, entries = load_index(store)
    by_target = {e.target: e for e in entries}
    out: list[dict] = []
    for p in proposals:
        if p.kind not in ("unique-term", "unique-term-swap"):
            continue
        e = by_target.get(p.target)
        if e is None or e.tagged:
            continue
        drop = [k for k in p.remove if k in e.keywords]
        keep = [k for k in e.keywords if k not in drop]
        add = [k for k in p.add if k not in keep]
        new_kws = keep + add
        if len(new_kws) < MIN_KEYWORDS or len(new_kws) > MAX_KEYWORDS:
            continue
        old_span = ",".join(e.keywords)
        new_span = ",".join(new_kws)
        if old_span == new_span:
            continue
        out.append({
            "file": p.target,
            "old_text": old_span,
            "new_text": new_span,
            "add": add,
            "remove": drop,
            "kind": p.kind,
        })
    return out


# ------------------------------------------------------------------ cli -------

def _load_session_messages(db: Path, session_id: str) -> list[dict]:
    """Minimal stdlib reader for the Hermes session db (no Hermes imports)."""
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT role, content, tool_calls FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        calls = r["tool_calls"]
        try:
            calls = json.loads(calls) if calls else []
        except (TypeError, ValueError):
            calls = []
        out.append({"role": r["role"], "content": r["content"], "tool_calls": calls})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="session-scoped memory keyword optimization")
    ap.add_argument("--store", default=str(Path.home() / ".hermes" / "memories"))
    ap.add_argument("--session", help="session id: include session-drift detector")
    ap.add_argument("--db", default=str(Path.home() / ".hermes" / "state.db"))
    ap.add_argument("--json", action="store_true", help="machine-readable proposals")
    ap.add_argument("--summary", action="store_true", help="print the one-line chat summary")
    ap.add_argument("--kinds", help="comma-separated detector filter")
    ap.add_argument("--limit", type=int, default=0, help="cap proposals per kind (0 = all)")
    ap.add_argument("--plan", action="store_true",
                    help="emit memory-tool edits (old_text/new_text) for the keyword proposals")
    ap.add_argument("--record", help="JSON: {file, add:[], remove:[]} already applied -> provenance")
    ap.add_argument("--max-files", type=int, default=0, help="apply plan for at most N files")
    args = ap.parse_args(argv)

    store = Path(os.path.expanduser(args.store))
    if not (store / INDEX_NAME).exists():
        print(f"ERROR: no index at {store / INDEX_NAME}", file=sys.stderr)
        return 2
    messages = None
    scope = None
    if args.session:
        messages = _load_session_messages(Path(os.path.expanduser(args.db)), args.session)
        scope = list(session_touched_files(messages, store).keys())
    kinds = args.kinds.split(",") if args.kinds else None
    props = analyze(store, messages=messages, scope=scope, kinds=kinds,
                    db=Path(os.path.expanduser(args.db)) if args.db else None)
    if args.limit:
        buckets: dict[str, list[Proposal]] = {}
        for p in props:
            buckets.setdefault(p.kind, []).append(p)
        props = [p for items in buckets.values() for p in items[: args.limit]]
    if args.record:
        rec = json.loads(args.record)
        record_applied(store, rec["file"], rec.get("add") or [], rec.get("remove") or [])
        print(f"recorded {len(rec.get('add') or [])} added / "
              f"{len(rec.get('remove') or [])} removed for {rec['file']}")
        return 0
    if args.plan:
        plan = apply_plan(store, props)
        if args.max_files:
            seen, capped = [], []
            for item in plan:
                if item["file"] not in seen:
                    if len(seen) >= args.max_files:
                        continue
                    seen.append(item["file"])
                capped.append(item)
            plan = capped
        print(json.dumps(plan, indent=1))
        return 0
    if args.json:
        print(json.dumps([p.as_dict() for p in props], indent=1))
    elif args.summary:
        print(summary_line(props))
    else:
        print(f"store: {store}  entries: {len(load_index(store)[1])}  proposals: {len(props)}")
        print(report(props))
        print("\n" + summary_line(props))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
