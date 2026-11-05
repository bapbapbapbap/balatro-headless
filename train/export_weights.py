"""Exports a training checkpoint as a small, model-only weights file for release.

Keeps the network weights (float32) and what is needed to use them and to know what they are:
vocabulary, architecture, action set and encoding, the reward it was trained on, the repo commit
and Balatro.exe hash it was trained against, its round and evaluation. Drops the optimizer,
retry pools and other training state. Contains no game code.

usage: python3 export_weights.py CHECKPOINT OUT.pt --version 0.3-beta [--note "..."]
Load with torch.load(path, weights_only=True); see play.py.
"""
import argparse
import datetime
import hashlib
import json
import os

import torch


PATH_KEYS = {"src", "out", "resume", "mirror_dir", "spill_dir", "bc", "sil_seed", "mods", "inject"}
# Paths to a released weights file or a checkpoint: kept as the file's name, which says what it was.
NAME_KEYS = {"moe_init", "moe_early", "init_weights"}


def public_provenance(prov):
    """Provenance without anything that identifies the machine or its owner: no host name, and
    local paths replaced by a placeholder. Training settings, commit and game hash stay."""
    prov = json.loads(json.dumps(prov, default=str))
    prov.pop("host", None)
    cfg = prov.get("config", {})
    for k in list(cfg):
        v = cfg[k]
        if k in PATH_KEYS and v:
            cfg[k] = [os.path.basename(str(x).rstrip("/")) for x in v] if isinstance(v, list) else "<local path>"
        elif k in NAME_KEYS and v:
            cfg[k] = [os.path.basename(str(x)) for x in v] if isinstance(v, list) else os.path.basename(str(v))
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--version", required=True, help="model version, e.g. 0.3-beta")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    prov = ck.get("provenance", {})
    arch = dict(ck.get("arch", {}))
    arch.setdefault("enc", 1)
    arch.setdefault("actions", "subsets")
    out = {
        "format": "balatro-headless-weights/1",
        "model_version": a.version,
        "exported_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": a.note,
        "state_dict": {k: v.detach().float().cpu().contiguous() for k, v in ck["model"].items()},
        "vocab": list(ck["vocab"]),
        "arch": arch,
        "trained": {
            "iteration": ck.get("iteration"),
            "best_eval_blinds": ck.get("best_eval"),
            "games": ck.get("games_total", ck.get("games")),
            "reward": prov.get("reward", ck.get("reward")),
            "provenance": public_provenance(prov),
        },
    }
    meta = json.dumps({k: v for k, v in out.items() if k != "state_dict"}, default=str)
    for leak in ("LAPTOP", "/home/", "/mnt/", "C:\\", "Users"):
        assert leak not in meta, f"exported metadata still contains '{leak}'"
    torch.save(out, a.out)
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    n = sum(v.numel() for v in out["state_dict"].values())
    print(json.dumps({"out": a.out, "sha256": h, "params": n, "arch": arch, "iteration": out["trained"]["iteration"],
                      "best_eval_blinds": out["trained"]["best_eval_blinds"]}, indent=1))


if __name__ == "__main__":
    main()
