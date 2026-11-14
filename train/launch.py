"""Starts (or resumes) a training run from its config, configs/<run>.toml.

    bash train/launch.sh <run> --work <work folder> [--smoke] [--hours H] [--workers N] [--skip-checks]
    python3 train/launch.py <run> --work <dir> --dry-run              print the train.py command, start nothing
    python3 train/launch.py <run> --work <dir> --check-provenance F   compare the config with a run's provenance.json

What a launch does, in order: the run's pre-flight checks (its config's `checks`, output in
<work>/runs/<run>/preflight.log; any failure stops the launch), then train.py with the config's settings, resuming
from runs/<run>/latest.pt when it exists, then (if the config says publish_on_stop and the run stopped on purpose)
publishes the run's new wins. Everything printed goes to runs/<run>/train.log as well.

A config has three tables: [run] (how to launch: start, hours, workers, checks, mirror_dir, spill_dir,
publish_on_stop), [train] (train.py's settings by name, as in train/options.py; booleans are flags), and [smoke]
(overrides for --smoke, which runs into runs/<run>-smoke from scratch). Paths may use {run} and {work}.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from options import build_parser  # noqa: E402

# Pre-flight checks by name: (what it shows, command, working dir, text the last line must contain).
# {src} is the game source; every command runs with this Python. A config can also give a check of its own as a
# table: {what = "...", cmd = ["tests/pin_check.py", "{src}", ...], cwd = "", want = "PASS"} (e.g. a hand pin's or
# a mixture of experts' check, which need the run's own weights and hand).
CHECKS = {
    "fixture-replay": ("a saved win replays exactly on the untouched game",
                       ["train/verify_games.py", "--src", "{src}", "tests/fixtures/win_v1_r549_1JA35NNR.jsonl",
                        "--wins-only", "--full", "--fresh-each"], "", "PASS"),
    "ordered": ("ordered actions do what they say (lead, Death, boss reroll)",
                ["tests/ordered_semantics.py", "{src}"], "", "ALL PASS"),
    "diff": ("fast vs untouched game, ordered actions + encoder 2",
             ["diff_check.py", "--src", "{src}", "--runs", "5", "--mode", "ordered", "--encode", "--enc", "2"],
             "harness", "PASS"),
    "v4-credit": ("routed credit", ["tests/v4_credit_check.py"], "", "ALL PASS"),
    "v4-phase": ("phase split: routing, replay, one network per decision, PPO",
                 ["tests/v4_phase_check.py", "{src}", "--games", "4", "--routes", "3"], "", "ALL PASS"),
    "v4-phi": ("phi_build and offer values: game unchanged, cache = fresh, offers = change",
               ["tests/v4_phi_check.py", "{src}", "--routes", "3"], "", "ALL PASS"),
    "v4-curriculum": ("curriculum: starts at the first shop of ante 4-6, own starts exact",
                      ["tests/v4_curriculum_check.py", "{src}", "--routes", "3"], "", "ALL PASS"),
    "v4c-swap": ("swap list: routes replay unchanged, swap = sell then buy, values = change, pool cards = shelf cards",
                 ["tests/v4c_swap_check.py", "{src}", "--routes", "3", "--swaps", "4"], "", "ALL PASS"),
    "clock": ("blind clock: skips, shaping and returns", ["tests/v3_check.py", "{src}", "--routes", "12"], "",
              "ALL PASS"),
    "repo": ("the repo's indexes, configs and map are consistent", ["tools/check_repo.py"], "", "ALL PASS"),
}
# set by the launcher, never by a config
OWNED = {"src", "run", "out", "hours", "workers", "resume", "init_weights", "warm_start", "mirror_dir", "spill_dir"}
# settings that only say where files go or how fast a machine is; ignored when comparing with a provenance
MACHINE = OWNED | {"device", "gpu_mem_fraction", "worker_mem_gb", "min_mem_gb", "min_disk_gb", "spill_at_gb"}


def load(run):
    path = os.path.join(REPO, "configs", f"{run}.toml")
    if not os.path.exists(path):
        raise SystemExit(f"no config for run '{run}': {path}")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    unknown = set(cfg) - {"run", "train", "smoke"}
    if unknown:
        raise SystemExit(f"{path}: unknown tables {sorted(unknown)}")
    return cfg


def to_argv(settings):
    """train.py arguments from a dict of settings (key = option name with underscores)."""
    known = {a.dest for a in build_parser()._actions}
    argv = []
    for k, v in settings.items():
        if k not in known:
            raise SystemExit(f"[train] {k}: train.py has no such setting (train/options.py)")
        if k in OWNED:
            raise SystemExit(f"[train] {k}: set by the launcher, not the config")
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False:
            continue
        elif isinstance(v, list):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv


def command(run, cfg, a, smoke):
    r, train = dict(cfg.get("run", {})), dict(cfg.get("train", {}))
    hours, workers = r.get("hours", 168), r.get("workers", 12)
    if smoke:
        sm = dict(cfg.get("smoke", {}))
        hours = sm.pop("hours", 1)
        train.update(sm)
    hours = a.hours if a.hours is not None else hours
    workers = a.workers if a.workers is not None else workers
    name = run + ("-smoke" if smoke else "")
    out = os.path.join(a.work, "runs", name)
    fill = lambda s: s.format(run=name, work=a.work)
    argv = [sys.executable, os.path.join(HERE, "train.py"), "--src", os.path.expanduser(a.src), "--run", name,
            "--out", out, "--hours", str(hours), "--workers", str(workers)]
    if r.get("mirror_dir"):
        argv += ["--mirror-dir", fill(r["mirror_dir"])]
    if r.get("spill_dir"):
        argv += ["--spill-dir", fill(r["spill_dir"])]
    start = r.get("start", "scratch")
    latest = os.path.join(out, "latest.pt")
    if os.path.exists(latest) and not smoke:
        argv += ["--resume", latest]
        how = f"resuming from {latest}"
    elif start == "scratch":
        how = "from scratch (random weights)"
    elif start.startswith("resume:"):
        argv += ["--resume", fill(start[7:])]
        how = f"from checkpoint {start[7:]}"
    elif start.startswith("weights:"):
        argv += ["--init-weights", fill(start[8:])]
        how = f"from released weights {start[8:]}"
    elif start.startswith("warm:"):
        argv += ["--warm-start", fill(start[5:])]
        how = f"warm start from checkpoint {fill(start[5:])} (fresh optimizer, round 0)"
    else:
        raise SystemExit(f"[run] start = {start!r}: use scratch, resume:<checkpoint>, weights:<file> or warm:<checkpoint>")
    return argv + to_argv(train), out, how, r


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")


def preflight(names, src, log, say):
    say("== pre-flight checks")
    with open(log, "w") as lf:
        for n in names:
            if isinstance(n, dict):
                what, cmd, cwd, want = n["what"], n["cmd"], n.get("cwd", ""), n["want"]
                n = what
            elif n in CHECKS:
                what, cmd, cwd, want = CHECKS[n]
            else:
                raise SystemExit(f"[run] checks: no check named '{n}' (launch.py CHECKS: {', '.join(CHECKS)})")
            cmd = [c.format(src=src) for c in cmd]
            wd = os.path.join(REPO, cwd)
            lf.write(f"== {n}: {' '.join(cmd)}\n")
            lf.flush()
            p = subprocess.run([sys.executable] + cmd, cwd=wd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
            lf.write(p.stdout)
            last = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
            if p.returncode != 0 or want not in last[0]:
                say(f"PRE-FLIGHT FAILED: {what} ({n}; see {log})")
                raise SystemExit(1)
            say(f"{what}: PASS")


def publish(run, out, say):
    """After a deliberate stop: replay the run's new wins and near wins, add the exact ones to wins/, update the
    tally, commit and push (as run_v2.sh did with PUBLISH_ON_STOP=1)."""
    try:
        status = json.load(open(os.path.join(out, "health.json"))).get("status", "")
    except (OSError, ValueError):
        status = ""
    if not status.startswith("stopped:") or status.startswith("stopped: crashed"):
        say(f"not publishing ({status or 'no status'})")
        return
    files = [p for p in (os.path.join(out, "wins.jsonl"), os.path.join(out, "near_wins.jsonl")) if os.path.exists(p)]
    say(f"== publishing {run}'s wins ({status})")
    t = os.path.join(REPO, "train")
    subprocess.run([sys.executable, "publish_wins.py", "--src", ARGS.src, "--out", "../wins"] + files, cwd=t)
    subprocess.run([sys.executable, "tally.py"], cwd=t, check=True)
    subprocess.run(["git", "-C", REPO, "add", "wins", "README.md"], check=True)
    if subprocess.run(["git", "-C", REPO, "diff", "--cached", "--quiet"]).returncode == 0:
        say("nothing new to publish")
        return
    subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", f"wins: publish {run}'s beaten seeds and update the tally"],
                   check=True)
    say("pushed" if subprocess.run(["git", "-C", REPO, "push", "-q"]).returncode == 0
        else f"committed but not pushed: run git push in {REPO}")


def check_provenance(argv, path):
    """Differences between the settings this config launches with and a run's provenance.json (machine and path
    settings ignored). Empty = the config reproduces that run."""
    parser = build_parser()
    have = vars(parser.parse_args(argv[2:]))
    ran = json.load(open(path))["config"]
    keys = (set(have) | set(ran)) - MACHINE
    # a setting added after that run (absent from its provenance) left at its default is the same run
    keys = {k for k in keys if k in ran or have.get(k) != parser.get_default(k)}
    return {k: (ran.get(k, "(absent)"), have.get(k, "(absent)")) for k in sorted(keys) if ran.get(k) != have.get(k)}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--work", default=os.environ.get("BALATRO_WORK"), help="work folder (or $BALATRO_WORK)")
    ap.add_argument("--src", default=os.path.expanduser(os.environ.get("BALATRO_SRC", "~/balatro/local/source")))
    ap.add_argument("--smoke", action="store_true", help="pre-flight, then a few rounds into runs/<run>-smoke")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--skip-checks", action="store_true", help="relaunch without the pre-flight checks")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-provenance", metavar="PROVENANCE_JSON")
    a = ARGS = ap.parse_args()
    if not a.work:
        raise SystemExit("--work <folder> (or set BALATRO_WORK)")
    cfg = load(a.run)
    argv, out, how, r = command(a.run, cfg, a, a.smoke)
    if a.check_provenance:
        d = check_provenance(argv, a.check_provenance)
        for k, (ran, now) in d.items():
            print(f"  {k}: ran with {ran!r}, config gives {now!r}")
        print("SAME SETTINGS" if not d else f"{len(d)} settings differ")
        return
    if a.dry_run:
        print(how)
        print(" ".join(argv))
        return
    if a.smoke:
        try:
            os.remove(os.path.join(out, "latest.pt"))
        except FileNotFoundError:
            pass
    os.makedirs(out, exist_ok=True)
    say = Tee(os.path.join(out, "train.log"))
    head = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
    say(f"== launch {a.run}{' (smoke)' if a.smoke else ''} {time.strftime('%Y-%m-%d %H:%M:%S')}  {head}")
    if not a.skip_checks:
        preflight(r.get("checks", []), a.src, os.path.join(out, "preflight.log"), say)
    say(f"== training {how}")
    say(" ".join(argv))
    # train.py's own output goes to train.log through this pipe, line by line
    p = subprocess.Popen(argv, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            say(line.rstrip("\n"))
    except KeyboardInterrupt:
        say("== Ctrl+C: waiting for train.py to save and stop")
    p.wait()
    say(f"== exited {time.strftime('%Y-%m-%d %H:%M:%S')} (code {p.returncode})")
    if r.get("publish_on_stop") and not a.smoke:
        publish(a.run, out, say)


if __name__ == "__main__":
    main()
