"""Prints the game's deck keys and challenge ids, one per line, from the pinned source.

usage: python3 tests/list_game.py --src ~/balatro/local/source [--decks | --challenges]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from env import BalatroEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
g = ap.add_mutually_exclusive_group()
g.add_argument("--decks", action="store_true")
g.add_argument("--challenges", action="store_true")
a = ap.parse_args()
env = BalatroEnv(a.src)
if not a.challenges:
    print("\n".join(env.deck_keys()))
if not a.decks:
    print("\n".join(env.challenge_ids()))
