#!/usr/bin/env python3
"""check_citations.py — detect stale `file.py:123` citations in the skill docs.

The skills anchor claims to specific source lines (`docs/AGENT_SKILLS.md`:
"Anchor every claim to a specific source file in this repo"). Those anchors rot
silently: any edit above a cited line shifts it, and the citation still *looks*
valid because it still points at a real line. Checking only for blank or
out-of-range lines misses exactly the cases that matter — a citation that has
drifted forty lines into unrelated code reads as fine.

So this asks a sharper question:

    does this citation still point at the same source text it pointed at when
    it was written?

For each citation it finds the commit that last touched the citing doc line
(`git blame`), reads the cited source line as of that commit, and compares it
with the line there now. Verdicts:

  ok        the cited line is byte-identical to what it was — nothing to do
  moved     that text now lives elsewhere in the file; --fix rewrites the number
  gone      the text is no longer anywhere in the file; a human must re-anchor
  range     the file is now shorter than the citation
  blank     stably points at a blank line — it pointed at one when written too,
            so this is pre-existing imprecision rather than drift. Reported but
            NOT treated as a failure, because nothing rotted.
  nobase    no baseline available (uncommitted doc line) — emptiness check only

Limitations worth knowing, because they bound what a green run proves:

  * A citation that was WRONG when written is reported `ok`. This verifies
    stability, not correctness. Only reading the claim against the code does
    that.
  * `git blame` returns whoever last touched the doc line, so reflowing prose
    around a citation moves its baseline forward and can mask drift that
    happened earlier.
  * Duplicate source lines (`)`, `else:`) relocate ambiguously; those are
    reported but never auto-fixed.
  * A bare `:NNN` inherits the file named earlier on its line, which is wrong
    when the line names a second file parenthetically — as one does while
    calling *another* doc's citation stale. Such a ref is checked against the
    wrong file; it stays quiet while that file is unchanged, and would report
    spurious drift if it changed. Add `citation-check: ignore` to the line if
    that happens.

Stdlib-only, like its siblings in this directory.

Usage:
    python3 check_citations.py                      # audit, exit 1 if anything rotted
    python3 check_citations.py --fix                # rewrite unambiguous `moved` ones
    python3 check_citations.py --json report.json   # machine-readable
    python3 check_citations.py --paths docs .claude # limit the scan
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# `path/to/file.py:123` or `:123-456`, with an optional directory prefix. The
# extension list is deliberate: bare `foo:12` in prose is not a citation.
CITATION_RE = re.compile(r"(?<![\w/.])([\w][\w./-]*\.(?:py|ya?ml|sh|toml|cfg))[:](\d+)(?:-(\d+))?(?![\w.])")

# These docs also use a continuation form -- `` (`:1819`) `` -- whose file is
# implied by a nearby full citation. There are enough of them that ignoring the
# form leaves a real blind spot, so they inherit the last full citation seen in
# the same paragraph (a run of non-blank lines).
BARE_RE = re.compile(r"`:(\d+)(?:-(\d+))?`")

# A file may also be named without a line number (``target_manager.py``) and
# then referred to by bare offsets. Such mentions set the inheritance context
# too, so they have to be tracked even though they are not citations.
FILEMENTION_RE = re.compile(r"(?<![\w/.])([\w][\w./-]*\.(?:py|ya?ml|sh|toml|cfg))(?![\w.])")

DOC_SUFFIXES = (".md", ".py", ".sh")
DEFAULT_PATHS = ("docs", ".claude", ".codex")
SUPPRESS = "citation-check: ignore"


def git(*args: str) -> str:
    """Run git and return stdout, or "" if the command fails."""
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def tracked_files(paths: tuple[str, ...]) -> list[str]:
    out = git("ls-files", *paths)
    return [p for p in out.split("\n") if p]


def resolve_source(name: str, tracked: list[str]) -> str | None:
    """Map a cited path fragment onto exactly one tracked file, else None."""
    if name in tracked:
        return name
    hits = [t for t in tracked if t.endswith("/" + name)]
    return hits[0] if len(hits) == 1 else None


def blame_commit(doc: str, line_no: int) -> str | None:
    """Commit that last touched this line of the doc."""
    out = git("blame", "-L", f"{line_no},{line_no}", "--porcelain", "HEAD", "--", doc)
    if not out:
        return None
    sha = out.split(" ", 1)[0].strip()
    # all-zero sha means the line is not committed yet
    return None if not sha or set(sha) <= {"0"} else sha


def file_at(rev: str, path: str) -> list[str] | None:
    out = git("show", f"{rev}:{path}")
    return out.splitlines() if out else None


def find_slice(hay: list[str], needle: list[str]) -> list[int]:
    """1-based start positions where `needle` occurs verbatim in `hay`."""
    if not needle:
        return []
    first, n = needle[0], len(needle)
    return [i + 1 for i in range(len(hay) - n + 1)
            if hay[i] == first and hay[i:i + n] == needle]


def dirty_docs() -> set[str]:
    """Docs with uncommitted changes, whose blame baseline cannot be trusted.

    `git blame` reports the last *committed* state of a line. If a citation was
    just rewritten but not yet committed, blame still names the old commit, and
    comparing the new line number against that old file content produces
    confident nonsense. So content-checking is skipped for these until the edit
    lands -- which is why --fix tells you to commit before re-running.
    """
    out = git("diff", "--name-only", "HEAD")
    return {p for p in out.split("\n") if p}


def classify(doc: str, doc_line: int, src: str, start: int, end: int,
             now: list[str], cache: dict[tuple[str, str], list[str] | None],
             baseline_ok: bool = True) -> dict[str, Any]:
    """One citation (a line or an inclusive range) -> a verdict dict.

    A range is compared as a whole slice rather than endpoint-by-endpoint. Doing
    it per-endpoint reports a range that merely *ends* on a blank line as broken,
    which is noise: the start line is what a reader jumps to, and the slice is
    what carries the claim.
    """
    cited = f"{start}" if start == end else f"{start}-{end}"
    base = {"doc": doc, "doc_line": doc_line, "source": src,
            "cited_line": start, "cited": cited}

    # Content first, emptiness second. A blank cited line is a *symptom* of
    # drift, and the content comparison says where the text actually went --
    # reporting "blank" instead throws that away. Ordering it this way also
    # keeps a mis-inherited bare ref quiet when the file it landed on has not
    # changed: there is genuinely no drift to report, however wrong the guess.
    def fallback() -> dict[str, Any] | None:
        if start > len(now):
            return {**base, "verdict": "range", "detail": f"file has {len(now)} lines"}
        if not now[start - 1].strip():
            return {**base, "verdict": "blank", "detail": "cited line is blank"}
        return None

    if not baseline_ok:
        return fallback() or {**base, "verdict": "nobase",
                              "detail": "doc has uncommitted changes; commit, then re-run"}

    sha = blame_commit(doc, doc_line)
    if sha is None:
        return fallback() or {**base, "verdict": "nobase",
                              "detail": "doc line not committed; only emptiness checked"}

    key = (sha, src)
    if key not in cache:
        cache[key] = file_at(sha, src)
    then = cache[key]
    if then is None or end > len(then):
        return fallback() or {**base, "verdict": "nobase",
                              "detail": f"source absent or shorter at {sha[:12]}"}

    want = then[start - 1:end]
    if now[start - 1:end] == want:
        if not want[0].strip():
            return {**base, "verdict": "blank",
                    "detail": "stably points at a blank line (cosmetic)"}
        return {**base, "verdict": "ok", "detail": want[0].strip()[:60]}

    if start > len(now):
        return {**base, "verdict": "range", "detail": f"file now has {len(now)} lines"}

    hits = find_slice(now, want)
    label = want[0].strip()[:48]
    if len(hits) == 1:
        shift = hits[0] - start
        return {**base, "verdict": "moved", "suggest": hits[0],
                "suggest_end": hits[0] + (end - start),
                "detail": f"{label!r} moved {shift:+d} to line {hits[0]}"}
    if len(hits) > 1:
        return {**base, "verdict": "moved", "candidates": hits,
                "detail": f"ambiguous: {label!r} matches at {hits[:6]}"}
    return {**base, "verdict": "gone", "detail": f"{label!r} no longer in the file"}


def scan(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    tracked = tracked_files(())
    docs = [d for d in tracked_files(paths) if d.endswith(DOC_SUFFIXES)]
    dirty = dirty_docs()
    now_cache: dict[str, list[str]] = {}
    hist_cache: dict[tuple[str, str], list[str] | None] = {}
    findings: list[dict[str, Any]] = []

    for doc in docs:
        try:
            lines = Path(doc).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        inherited: str | None = None      # last full citation in this paragraph
        for i, text in enumerate(lines, start=1):
            if not text.strip():
                inherited = None          # paragraph break ends inheritance
                continue
            if SUPPRESS in text:
                continue

            # Walk the line left to right so a bare `:NNN` inherits the file
            # named *before* it, never one named later. Getting this wrong makes
            # a bare offset resolve against an unrelated file and report a
            # confident out-of-range error.
            events: list[tuple[int, str, tuple[str, ...]]] = []
            spans: list[tuple[int, int]] = []
            for m in CITATION_RE.finditer(text):
                events.append((m.start(), "cite", m.groups()))
                spans.append(m.span())
            for m in FILEMENTION_RE.finditer(text):
                # skip the filename halves of full citations already recorded
                if any(lo <= m.start() < hi for lo, hi in spans):
                    continue
                events.append((m.start(), "mention", m.groups()))
            for m in BARE_RE.finditer(text):
                events.append((m.start(), "bare", m.groups()))
            events.sort(key=lambda e: e[0])

            refs: list[tuple[str, str, str, bool]] = []
            for _pos, kind, groups in events:
                if kind == "mention":
                    src = resolve_source(groups[0], tracked)
                    if src is not None and src != doc:
                        inherited = src
                elif kind == "cite":
                    name, a, b = groups
                    src = resolve_source(name, tracked)
                    if src is not None and src != doc:
                        refs.append((src, a, b or "", False))
                        inherited = src
                elif inherited is not None:
                    a, b = groups
                    refs.append((inherited, a, b or "", True))

            for src, a, b, is_bare in refs:
                if src not in now_cache:
                    try:
                        now_cache[src] = Path(src).read_text(encoding="utf-8").splitlines()
                    except (OSError, UnicodeDecodeError):
                        now_cache[src] = []
                start = int(a)
                end = int(b) if b else start
                if end < start:
                    end = start
                f = classify(doc, i, src, start, end, now_cache[src], hist_cache,
                             baseline_ok=doc not in dirty)
                if is_bare:
                    f["inherited_source"] = True
                findings.append(f)
    return findings


def apply_fixes(findings: list[dict[str, Any]]) -> int:
    """Rewrite unambiguous `moved` citations in place."""
    per_doc: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        if f["verdict"] == "moved" and "suggest" in f:
            per_doc.setdefault(f["doc"], []).append(f)
    fixed = 0
    for doc, items in per_doc.items():
        p = Path(doc)
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        for f in items:
            idx = f["doc_line"] - 1
            base = f["source"].rsplit("/", 1)[-1]
            # Rewrite only the number, and only next to this file's basename, so
            # a doc citing two files on one line cannot cross-contaminate.
            if f.get("inherited_source"):
                # No filename beside a bare `:NNN`, so anchor on the backticks.
                pat = re.compile(rf"(`:){re.escape(f['cited'])}(?=`)")
            else:
                pat = re.compile(rf"({re.escape(base)}:){re.escape(f['cited'])}(?![\d-])")
            repl = (str(f["suggest"]) if f["suggest"] == f["suggest_end"]
                    else f"{f['suggest']}-{f['suggest_end']}")
            new, n = pat.subn(rf"\g<1>{repl}", lines[idx])
            if n:
                lines[idx] = new
                fixed += n
        p.write_text("".join(lines), encoding="utf-8")
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*", default=list(DEFAULT_PATHS), help="paths to scan")
    ap.add_argument("--fix", action="store_true", help="rewrite unambiguous `moved` citations")
    ap.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    if not git("rev-parse", "--git-dir"):
        print("not a git repository", file=sys.stderr)
        return 2

    findings = scan(tuple(args.paths))
    counts = Counter(f["verdict"] for f in findings)
    # `blank` is informational: content-first ordering means the citation pointed
    # at a blank line when it was written, so there is no drift to act on.
    problems = [f for f in findings if f["verdict"] not in ("ok", "nobase", "blank")]
    notes = [f for f in findings if f["verdict"] == "blank"]

    if not args.quiet:
        for f in sorted(problems + notes, key=lambda f: (f["verdict"], f["doc"], f["doc_line"])):
            tag = " (inherited)" if f.get("inherited_source") else ""
            print(f"{f['verdict'].upper():6s} {f['doc']}:{f['doc_line']}  "
                  f"-> {f['source']}:{f['cited']}{tag}\n       {f['detail']}")

    if args.json:
        Path(args.json).write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.json}")

    total = len(findings)
    print(f"\n{total} citations checked: " +
          ", ".join(f"{n} {v}" for v, n in sorted(counts.items())) or "none found")

    if args.fix:
        n = apply_fixes(findings)
        print(f"rewrote {n} citation(s) -- COMMIT, then re-run to confirm.\n"
              "Re-running against uncommitted edits cannot content-check them: blame\n"
              "still names the pre-edit commit, so the new numbers would be compared\n"
              "against the old file.")
        return 0 if n else (1 if problems else 0)

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
