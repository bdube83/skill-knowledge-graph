"""Larger-context corpus generator for the H1 token-reduction hypothesis.

The original eval corpus (eval/corpus.jsonl) carries tiny per-task contexts
(median LLM input around 90 tokens), which sit below the SKG routing-header
cost (around 120 tokens). On that corpus, SKG looks worse than LLM-only on
input tokens. This generator builds a 200-task corpus where each task's
context body is large enough (500 to 1500 tokens of plausible content) that
SKG hits should pay back the header cost on average.

Schema matches eval/corpus.jsonl. Five categories, 40 tasks per category:

  - communication: PR diff, comment thread, reviewer history.
  - documentation: markdown section, surrounding sections, style guide.
  - git:           git log, branch list, recent diff.
  - planning:      issue triage list, sprint capacity table.
  - analysis:      code metrics, error log excerpt, dependency tree.

The content is synthetic. No customer or employer data is included.
Generation uses random.Random(42) for determinism.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


CATEGORIES         = ["communication", "documentation", "git", "planning", "analysis"]
TASKS_PER_CATEGORY = 40


# Synthetic vocabulary used to assemble plausible content. Names are common
# placeholders; modules and identifiers are generic engineering vocabulary.
PEOPLE   = ["alice", "bob", "carol", "dan", "erin", "frank", "grace", "hugo", "ivy", "jules"]
MODULES  = ["parser", "router", "scheduler", "store", "indexer", "verifier", "launcher", "registry", "queue", "cache"]
VERBS    = ["fix", "add", "refactor", "update", "rename", "remove", "tighten", "split", "merge", "extract"]
TOPICS   = ["timeout handling", "retry policy", "config loader", "schema check", "auth header", "log redaction",
            "rate limit", "error code map", "metrics emitter", "session token"]
FILES    = ["src/{m}.py", "src/{m}_test.py", "internal/{m}/handler.go", "lib/{m}/index.ts", "pkg/{m}/state.rs"]
ERRORS   = ["TimeoutError", "ValueError", "KeyError", "ConnectionResetError", "PermissionError",
            "ParseError", "SchemaMismatch", "MissingField", "RateLimited", "AuthFailed"]


def _pick(rng: random.Random, items: list) -> str:
    return rng.choice(items)


def _filename(rng: random.Random) -> str:
    tmpl = rng.choice(FILES)
    return tmpl.replace("{m}", rng.choice(MODULES))


# -----------------------------------------------------------------------------
# Communication: PR diff + comment thread + reviewer history
# -----------------------------------------------------------------------------

def _fake_diff(rng: random.Random) -> str:
    lines = []
    n_files = rng.randint(1, 3)
    for _ in range(n_files):
        path = _filename(rng)
        lines.append(f"diff --git a/{path} b/{path}")
        lines.append(f"--- a/{path}")
        lines.append(f"+++ b/{path}")
        n_hunks = rng.randint(1, 2)
        for _ in range(n_hunks):
            start_a = rng.randint(10, 400)
            start_b = start_a + rng.randint(-2, 4)
            len_a   = rng.randint(3, 12)
            len_b   = len_a + rng.randint(-2, 5)
            lines.append(f"@@ -{start_a},{len_a} +{start_b},{len_b} @@ def {_pick(rng, MODULES)}_{_pick(rng, VERBS)}():")
            n_chunk = rng.randint(6, 14)
            for _ in range(n_chunk):
                marker = rng.choice([" ", " ", "+", "-"])
                ident  = _pick(rng, MODULES)
                verb   = _pick(rng, VERBS)
                topic  = _pick(rng, TOPICS)
                snippet_choices = [
                    f"if {ident}.state == 'ready':",
                    f"    return {ident}.{verb}({rng.randint(1, 99)})",
                    f"raise {_pick(rng, ERRORS)}('{topic}')",
                    f"# TODO: {verb} {topic} before release",
                    f"self._{ident} = {ident}_factory()",
                    f"logger.info('handled %s in %dms', '{topic}', {rng.randint(1, 250)})",
                    f"value = config.get('{topic.replace(' ', '_')}', None)",
                    f"assert value is not None, 'missing {topic}'",
                ]
                lines.append(f"{marker}{rng.choice(snippet_choices)}")
    return "\n".join(lines)


def _fake_thread(rng: random.Random) -> list[dict]:
    n = rng.randint(4, 8)
    out = []
    for _ in range(n):
        author = _pick(rng, PEOPLE)
        body_parts = []
        for _ in range(rng.randint(1, 3)):
            body_parts.append(rng.choice([
                f"I think the {_pick(rng, MODULES)} change covers {_pick(rng, TOPICS)}.",
                f"Can you split the {_pick(rng, VERBS)} of {_pick(rng, TOPICS)} into a separate commit?",
                f"The new test for {_pick(rng, ERRORS)} reads cleanly.",
                f"Watch out for the {_pick(rng, TOPICS)} edge case when input is empty.",
                f"Style nit: rename `{_pick(rng, MODULES)}_x` to `{_pick(rng, MODULES)}_state`.",
                f"This block duplicates {_pick(rng, MODULES)}.handle; pull into a helper.",
                f"I would prefer we land {_pick(rng, TOPICS)} in a follow-up PR.",
                f"Coverage on {_pick(rng, MODULES)}.py looks good after this change.",
            ]))
        out.append({"author": author, "body": " ".join(body_parts)})
    return out


def _reviewer_history(rng: random.Random) -> list[dict]:
    n = rng.randint(2, 4)
    out = []
    for _ in range(n):
        out.append({
            "reviewer":     _pick(rng, PEOPLE),
            "prs_reviewed": rng.randint(5, 80),
            "median_turnaround_hours": round(rng.uniform(2.0, 48.0), 1),
            "areas":        rng.sample(MODULES, k=rng.randint(1, 3)),
        })
    return out


def _make_communication(rng: random.Random, idx: int) -> dict:
    task_choices = [
        "Draft a reviewer ping that summarises the diff and asks for review",
        "Compose a follow-up message for a stalled review thread",
        "Write a release-note blurb describing the PR scope",
        "Draft a comment that addresses the open review feedback",
        "Write a tag-in message asking another reviewer to weigh in",
    ]
    return {
        "task":     rng.choice(task_choices),
        "category": "communication",
        "context":  {
            "pr_number":         str(idx),
            "repo":              "example/repo",
            "author":            _pick(rng, PEOPLE),
            "reviewers":         rng.sample(PEOPLE, k=rng.randint(1, 3)),
            "diff":              _fake_diff(rng),
            "comments":          _fake_thread(rng),
            "reviewer_history":  _reviewer_history(rng),
        },
    }


# -----------------------------------------------------------------------------
# Documentation: markdown section + surrounding context + style guide excerpt
# -----------------------------------------------------------------------------

def _fake_markdown_section(rng: random.Random, words_low: int, words_high: int) -> str:
    target = rng.randint(words_low, words_high)
    parts  = []
    sentences = [
        "The {m} accepts a config dict with {n} required keys and emits structured logs.",
        "Callers should treat {m} as the single owner of {topic} state.",
        "Errors of type {err} bubble up unchanged so the host can decide retry policy.",
        "Set {m}.timeout to a positive integer to bound long-running calls.",
        "The default value for {topic} is chosen to match the {m} reference implementation.",
        "Use the {m}.metrics counter to track {topic} per request.",
        "Avoid passing closures into {m}; pass plain dicts so the runtime can serialise them.",
        "When {topic} is empty, {m} short-circuits and returns the zero value.",
        "The {m} integrates with the platform metrics layer through a thin adapter.",
        "Provide a unique request id in the header so {m} can correlate logs across hops.",
        "The retry path runs at most {n} times before surfacing {err}.",
        "Read the policy file once at startup; {m} caches the parsed form.",
    ]
    written = 0
    while written < target:
        s = rng.choice(sentences)
        s = s.replace("{m}",     _pick(rng, MODULES))
        s = s.replace("{topic}", _pick(rng, TOPICS))
        s = s.replace("{err}",   _pick(rng, ERRORS))
        s = s.replace("{n}",     str(rng.randint(2, 9)))
        parts.append(s)
        written += len(s.split())
    return " ".join(parts)


def _surrounding_sections(rng: random.Random) -> list[dict]:
    titles = ["Overview", "Configuration", "Error handling", "Metrics", "Examples", "Migration notes"]
    chosen = rng.sample(titles, k=rng.randint(2, 3))
    return [{"title": t, "body": _fake_markdown_section(rng, 30, 80)} for t in chosen]


def _style_guide_excerpt(rng: random.Random) -> str:
    rules = [
        "Lead with the conclusion; supporting detail comes after.",
        "Use active voice with concrete subjects.",
        "Spell out acronyms on first use within a section.",
        "Use code spans for identifiers and config keys.",
        "Prefer one idea per sentence.",
        "Capitalise product names; lowercase generic nouns.",
        "Cross-link to the configuration reference rather than restating fields.",
        "Use British English in prose; identifiers stay as written in code.",
        "Avoid hedging adverbs in the introduction.",
        "Mark deprecated APIs with a callout block.",
    ]
    return " ".join(rng.sample(rules, k=rng.randint(4, 7)))


def _make_documentation(rng: random.Random, idx: int) -> dict:
    task_choices = [
        "Update the section to reflect the new config keys",
        "Tighten the section to match the style guide",
        "Add an examples block to the section",
        "Rewrite the error-handling subsection for clarity",
        "Add a migration note that points to the new endpoint",
    ]
    return {
        "task":     rng.choice(task_choices),
        "category": "documentation",
        "context":  {
            "module":    _pick(rng, MODULES),
            "section":   _fake_markdown_section(rng, 180, 360),
            "neighbours": _surrounding_sections(rng),
            "style_guide": _style_guide_excerpt(rng),
        },
    }


# -----------------------------------------------------------------------------
# Git: log + branches + recent diff
# -----------------------------------------------------------------------------

def _fake_git_log(rng: random.Random) -> list[dict]:
    n = rng.randint(15, 30)
    out = []
    for _ in range(n):
        sha    = f"{rng.randrange(0, 0xffffffff):08x}"
        author = _pick(rng, PEOPLE)
        verb   = _pick(rng, VERBS)
        topic  = _pick(rng, TOPICS)
        ident  = _pick(rng, MODULES)
        msg    = f"{verb}({ident}): {topic} cleanup"
        out.append({"sha": sha, "author": author, "message": msg, "files_changed": rng.randint(1, 8)})
    return out


def _fake_branches(rng: random.Random) -> list[dict]:
    n = rng.randint(6, 12)
    out = []
    for _ in range(n):
        out.append({
            "name":   f"{_pick(rng, VERBS)}-{_pick(rng, MODULES)}-{rng.randint(1, 999)}",
            "ahead":  rng.randint(0, 30),
            "behind": rng.randint(0, 80),
            "author": _pick(rng, PEOPLE),
        })
    return out


def _make_git(rng: random.Random, idx: int) -> dict:
    task_choices = [
        "Summarise the recent git log into a one-paragraph status",
        "Identify branches that are far behind main and likely stale",
        "Group recent commits by area and describe the trend",
        "Draft a release note from the recent commit history",
        "Find PRs that look risky based on diff size and history",
    ]
    return {
        "task":     rng.choice(task_choices),
        "category": "git",
        "context":  {
            "repo":     "example/repo",
            "log":      _fake_git_log(rng),
            "branches": _fake_branches(rng),
            "recent_diff": _fake_diff(rng),
        },
    }


# -----------------------------------------------------------------------------
# Planning: issue triage list + sprint capacity table
# -----------------------------------------------------------------------------

def _fake_issues(rng: random.Random) -> list[dict]:
    n = rng.randint(8, 16)
    out = []
    severities = ["low", "medium", "high", "critical"]
    for i in range(n):
        body_sentences = []
        for _ in range(rng.randint(1, 3)):
            body_sentences.append(rng.choice([
                f"Users report {_pick(rng, ERRORS)} when calling {_pick(rng, MODULES)}.{_pick(rng, VERBS)}.",
                f"The {_pick(rng, TOPICS)} flow drops requests above {rng.randint(50, 500)} concurrent calls.",
                f"Logs show repeated {_pick(rng, ERRORS)} from the {_pick(rng, MODULES)} layer.",
                f"Reproduces consistently on staging; intermittent in production.",
                f"Suspect interaction between {_pick(rng, MODULES)} and {_pick(rng, MODULES)}.",
                f"Workaround: pin the {_pick(rng, TOPICS)} value to a stable default.",
            ]))
        out.append({
            "issue":    rng.randint(100, 9999),
            "title":    f"{_pick(rng, MODULES)}: {_pick(rng, TOPICS)} regression",
            "severity": rng.choice(severities),
            "body":     " ".join(body_sentences),
            "labels":   rng.sample(["bug", "perf", "ux", "infra", "docs", "test"], k=rng.randint(1, 3)),
        })
    return out


def _fake_capacity(rng: random.Random) -> list[dict]:
    out = []
    for p in rng.sample(PEOPLE, k=rng.randint(4, 7)):
        out.append({
            "person":      p,
            "capacity_pts": rng.randint(3, 13),
            "carryover":   rng.randint(0, 5),
            "focus":       _pick(rng, MODULES),
        })
    return out


def _make_planning(rng: random.Random, idx: int) -> dict:
    task_choices = [
        "Triage the issue list and propose a sprint plan",
        "Group issues by area and pick the top three for the next sprint",
        "Estimate effort across the issue list and flag risky items",
        "Draft a sprint plan that respects the capacity table",
        "Identify issues that can be deferred and explain why",
    ]
    return {
        "task":     rng.choice(task_choices),
        "category": "planning",
        "context":  {
            "sprint":   rng.randint(1, 50),
            "issues":   _fake_issues(rng),
            "capacity": _fake_capacity(rng),
        },
    }


# -----------------------------------------------------------------------------
# Analysis: metrics dump + error log + dependency tree
# -----------------------------------------------------------------------------

def _fake_metrics(rng: random.Random) -> dict:
    files = []
    for _ in range(rng.randint(6, 12)):
        files.append({
            "path":     _filename(rng),
            "loc":      rng.randint(80, 2000),
            "covered":  round(rng.uniform(0.4, 0.99), 3),
            "complexity_p95": rng.randint(3, 28),
        })
    return {
        "files":           files,
        "total_loc":       sum(f["loc"] for f in files),
        "weighted_coverage": round(rng.uniform(0.55, 0.92), 3),
        "test_runtime_s":  round(rng.uniform(8.0, 240.0), 2),
    }


def _fake_error_log(rng: random.Random) -> str:
    n = rng.randint(30, 90)
    lines = []
    for _ in range(n):
        ts = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        level = rng.choices(["INFO", "WARN", "ERROR"], weights=[2, 3, 5])[0]
        err   = _pick(rng, ERRORS)
        mod   = _pick(rng, MODULES)
        topic = _pick(rng, TOPICS)
        lines.append(f"{ts} {level} {mod}: {err} during {topic} (request_id={rng.randrange(0, 0xffffff):06x})")
    return "\n".join(lines)


def _fake_dependency_tree(rng: random.Random) -> dict:
    deps = {}
    for _ in range(rng.randint(6, 10)):
        name = f"{_pick(rng, MODULES)}-{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 50)}"
        sub  = []
        for _ in range(rng.randint(0, 3)):
            sub.append(f"{_pick(rng, MODULES)}-{rng.randint(0, 9)}.{rng.randint(0, 20)}")
        deps[name] = sub
    return deps


def _make_analysis(rng: random.Random, idx: int) -> dict:
    task_choices = [
        "Identify the most common error pattern from the log excerpt",
        "Summarise code metrics and flag files that need attention",
        "Compare error frequency across modules and rank the top three",
        "Pick out unusual dependency versions from the tree",
        "Draft a one-paragraph reliability summary for the period",
    ]
    return {
        "task":     rng.choice(task_choices),
        "category": "analysis",
        "context":  {
            "metrics":         _fake_metrics(rng),
            "error_log":       _fake_error_log(rng),
            "dependency_tree": _fake_dependency_tree(rng),
        },
    }


# -----------------------------------------------------------------------------
# Top-level corpus assembly
# -----------------------------------------------------------------------------

CATEGORY_BUILDERS = {
    "communication": _make_communication,
    "documentation": _make_documentation,
    "git":           _make_git,
    "planning":      _make_planning,
    "analysis":      _make_analysis,
}


def generate_large_corpus(seed: int = 42) -> list[dict]:
    """Generate a 200-task corpus with 40 tasks per category.

    Output is a list of dicts in the same shape as eval/corpus.jsonl. The
    seed is fixed by default so calling this twice with the same argument
    yields byte-identical output.
    """
    rng = random.Random(seed)
    tasks: list[dict] = []
    idx = 1
    for category in CATEGORIES:
        builder = CATEGORY_BUILDERS[category]
        for _ in range(TASKS_PER_CATEGORY):
            t = builder(rng, idx)
            t["id"] = f"t{idx:04d}"
            t["expected_stage"] = "miss"
            # Reorder keys to match the existing corpus.jsonl shape.
            tasks.append({
                "id":             t["id"],
                "task":           t["task"],
                "category":       t["category"],
                "context":        t["context"],
                "expected_stage": t["expected_stage"],
            })
            idx += 1
    return tasks


def save_corpus(corpus: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in corpus:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the larger-context SKG eval corpus.")
    parser.add_argument("--out",  default="eval/corpus_large.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = generate_large_corpus(seed=args.seed)
    out    = Path(args.out)
    save_corpus(corpus, out)

    cats = {}
    for t in corpus:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    body_chars = [len(json.dumps(t["context"])) for t in corpus]
    print(f"Generated {len(corpus)} tasks -> {out}")
    print(f"Category counts: {cats}")
    print(f"Context body chars: min={min(body_chars)} median={sorted(body_chars)[len(body_chars)//2]} max={max(body_chars)}")
