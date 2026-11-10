"""The worker pool and the safety exceptions training stops on."""

import os
import queue
import shutil
import sys
import time

import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rollout import worker_main  # noqa: E402


# ---------------- safety ----------------

class Stop(Exception):
    """Ends training cleanly: weights are saved and the reason recorded."""

class WorkerFailure(Exception):
    pass


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("inf")


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 2**30


class Pool:
    """The worker processes. Restartable: a crashed or hung worker gets the whole pool restarted."""

    def __init__(self, args):
        self.args, self.ctx, self.gen, self.procs = args, mp.get_context("spawn"), -1, []

    def start(self, timeout=600):
        self.gen += 1
        self.stash = {}
        W = self.args.workers
        self.in_qs = [self.ctx.Queue() for _ in range(W)]
        self.out_q = self.ctx.Queue()
        self.procs = [self.ctx.Process(target=worker_main, args=(w, self.args, self.in_qs[w], self.out_q), daemon=True)
                      for w in range(W)]
        for p in self.procs:
            p.start()
        vocab, deadline = None, time.time() + timeout
        for _ in range(W):
            try:
                tag, wid, v = self.out_q.get(timeout=max(1, deadline - time.time()))
            except queue.Empty:
                raise WorkerFailure("workers did not start in time")
            if tag == "error":
                raise WorkerFailure(f"worker {wid} failed to start:\n{v}")
            vocab = vocab or v
            if v != vocab:
                raise WorkerFailure("workers disagree on vocabulary")
        self.vocab = vocab

    def stop(self):
        for q in getattr(self, "in_qs", []):
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.procs = []

    # Requests are numbered (tickets), so a round can be sent, left running while the learner works,
    # and collected later; replies to other requests that arrive meanwhile are kept for them.
    def send(self, state, seeds_per_worker, mode, it):
        self.ticket = getattr(self, "ticket", 0) + 1
        for w in range(self.args.workers):
            self.in_qs[w].put((state, seeds_per_worker[w], mode, it, self.gen, self.ticket))
        return (self.gen, self.ticket)

    def collect(self, ticket, timeout, should_stop=lambda: None):
        gen, t = ticket
        if gen != self.gen:
            raise WorkerFailure("workers were restarted after this round was sent")
        got = self.stash.pop(t, [])
        deadline = time.time() + timeout
        while len(got) < self.args.workers:
            why = should_stop()               # wait in 1 s slices so a stop request is seen at once
            if why:
                raise Stop(why)
            try:
                tag, wid, e, rt = self.out_q.get(timeout=1)
            except queue.Empty:
                if time.time() > deadline:
                    dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
                    raise WorkerFailure(f"round timed out after {timeout}s (dead workers: {dead or 'none, hung'})")
                continue
            if rt == t:
                got.append((tag, wid, e))
            else:
                self.stash.setdefault(rt, []).append((tag, wid, e))
        errors = [f"worker {wid}:\n{e}" for tag, wid, e in got if tag == "error"]
        if errors:
            raise WorkerFailure("\n".join(errors))
        return [x for tag, wid, e in got for x in e]

    def play(self, state, seeds_per_worker, mode, it, timeout, should_stop=lambda: None):
        return self.collect(self.send(state, seeds_per_worker, mode, it), timeout, should_stop)
