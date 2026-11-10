"""Keeps the repo's maps and indexes true. Prints ALL PASS or every problem; --fix adds missing index rows.

    python3 tools/check_repo.py [--fix]

Checks:
1. docs/hypotheses.md: its file table lists every docs/hypotheses-*.md, and its section table has one row per
   "## " section in those files, each linking to the right file and anchor (GitHub's anchor rules).
2. docs/runs/README.md lists every run page in docs/runs/.
3. Every configs/<run>.toml loads, sets only train.py's own settings, names known pre-flight checks, and has a run
   page docs/runs/<run>.md.
4. Every relative link in AGENTS.md, ARCHIVE.md, README.md, llms.txt and docs/ (the hypothesis record aside) points
   at a file that exists, and so does every backticked repo path in AGENTS.md, README.md and llms.txt (skipped:
   <placeholders>, globs, and lines that say the file is at a tag or in ARCHIVE.md).
Runs in every launch's pre-flight (check "repo"), so a map can't drift from the tree unnoticed.
"""
import argparse
import glob
import os
import re
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))


def slug(heading):
    """GitHub's anchor for a markdown heading."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def rel(p):
    return os.path.relpath(p, REPO)


def check_hypotheses(fix, problems):
    idx_path = os.path.join(REPO, "docs", "hypotheses.md")
    idx = open(idx_path).read()
    files = sorted(glob.glob(os.path.join(REPO, "docs", "hypotheses-*.md")))
    for f in files:
        name = os.path.basename(f)
        if f"(`{name}`)" not in idx.replace(f"[`{name}`]({name})", f"(`{name}`)"):
            problems.append(f"docs/hypotheses.md: file table has no row for {name}")
    rows = re.findall(r"^\| \[(.+?)\]\((hypotheses-[\w-]+\.md)#([^)]+)\) \| ([\w-]+) \|$", idx, re.M)
    have = {(f, a) for _, f, a, _ in rows}
    missing = []
    for f in files:
        name = os.path.basename(f)
        heads = [h for h in re.findall(r"^## (.+)$", open(f).read(), re.M)]
        anchors = {slug(h) for h in heads}
        for title, rf, a, _ in rows:
            if rf == name and a not in anchors:
                problems.append(f"docs/hypotheses.md: row '{title[:60]}' links to a missing anchor {name}#{a}")
        for h in heads:
            if (name, slug(h)) not in have:
                missing.append((h, name))
    for h, name in missing:
        run = name[len("hypotheses-"):-3]
        row = f"| [{h}]({name}#{slug(h)}) | {run} |"
        if fix:
            idx = idx.rstrip("\n") + "\n" + row + "\n"
        else:
            problems.append(f"docs/hypotheses.md: no row for '{h[:70]}' ({name}); --fix adds it")
    if fix and missing:
        open(idx_path, "w").write(idx)
        print(f"added {len(missing)} rows to docs/hypotheses.md")


def check_runs(problems):
    readme = open(os.path.join(REPO, "docs", "runs", "README.md")).read()
    for f in sorted(glob.glob(os.path.join(REPO, "docs", "runs", "*.md"))):
        name = os.path.basename(f)
        if name != "README.md" and f"]({name})" not in readme:
            problems.append(f"docs/runs/README.md: no row for {name}")


def check_configs(problems):
    from options import build_parser
    import launch
    known = {a.dest for a in build_parser()._actions}
    for f in sorted(glob.glob(os.path.join(REPO, "configs", "*.toml"))):
        run = os.path.basename(f)[:-5]
        try:
            cfg = tomllib.load(open(f, "rb"))
        except tomllib.TOMLDecodeError as e:
            problems.append(f"{rel(f)}: not valid TOML ({e})")
            continue
        for k in set(cfg) - {"run", "train", "smoke"}:
            problems.append(f"{rel(f)}: unknown table [{k}]")
        for table in ("train", "smoke"):
            for k in cfg.get(table, {}):
                if k == "hours" and table == "smoke":
                    continue
                if k not in known:
                    problems.append(f"{rel(f)}: [{table}] {k} is not a train.py setting")
                elif k in launch.OWNED:
                    problems.append(f"{rel(f)}: [{table}] {k} is set by the launcher")
        for c in cfg.get("run", {}).get("checks", []):
            if isinstance(c, str) and c not in launch.CHECKS:
                problems.append(f"{rel(f)}: check '{c}' is not in launch.py's CHECKS")
        if not os.path.exists(os.path.join(REPO, "docs", "runs", f"{run}.md")):
            problems.append(f"{rel(f)}: no run page docs/runs/{run}.md")


PATHISH = re.compile(r"`((?:train|harness|tests|tools|analysis|docs|configs|naneinf|wins|weights)/[\w./-]+)`")
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def check_paths(problems):
    docs = ["AGENTS.md", "ARCHIVE.md", "README.md", "llms.txt"] + [rel(p) for p in glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True)]
    for d in docs:
        path = os.path.join(REPO, d)
        if not os.path.exists(path):
            problems.append(f"{d}: missing")
            continue
        text = open(path).read()
        if os.path.basename(d).startswith("hypotheses-") or d == "docs/closed_hypotheses.md":
            continue               # the record: paths there are as they were when written (ARCHIVE.md has old ones)
        base = os.path.dirname(path)
        for m in LINK.finditer(text):
            t = m.group(1)
            if "://" in t or t.startswith("mailto:") or "<" in t:
                continue
            if not os.path.exists(os.path.normpath(os.path.join(base, t))):
                problems.append(f"{d}: link to missing {t}")
        if d in ("AGENTS.md", "llms.txt", "README.md"):
            for line in text.splitlines():
                if "ARCHIVE.md" in line or "at tag" in line:
                    continue          # names a file kept only in a tagged commit
                for m in PATHISH.finditer(line):
                    t = m.group(1).rstrip(".")
                    if any(c in t for c in "<*{") or t.endswith("/"):
                        continue
                    if not os.path.exists(os.path.join(REPO, t)):
                        problems.append(f"{d}: `{t}` does not exist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="add missing rows to docs/hypotheses.md")
    a = ap.parse_args()
    problems = []
    check_hypotheses(a.fix, problems)
    check_runs(problems)
    check_configs(problems)
    check_paths(problems)
    print("\n".join(problems) if problems else "ALL PASS")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
