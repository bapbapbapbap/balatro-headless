# Test T (docs/hypotheses-v6.md, 2026-11-21 23:42 section). Usage: python3 test_t_teacher_on_deaths.py <eval_ladder output> <n>
# Paths below are the cloud session's (Balatro Lab clone, game source); edit for the laptop.
"""Test T (docs/hypotheses-v6.md, 2026-11-21 23:42): the teacher's move on every state of v6e's temperature-0
ante-1 losses, replayed move for move through teach.py's game loop (the shadow state stays in step)."""
import json, os, sys
sys.path.insert(0, '/home/claude/balatro-headless/reproduce/balatro_lab')
os.environ.setdefault("VALUEBOT_WEIGHTS", "hand")
import teach as T

class Replayer:
    path, iteration = "replay of v6e t0", None
    def __init__(self, acts): self.acts = acts
    def reset(self, seed): self.i = 0
    def move(self, o, member):
        a = self.acts[self.i]; self.i += 1; return a

games = {g['seed']: g for g in map(json.loads, list(open(sys.argv[1]))[1:])}
seeds = sorted(s for s, g in games.items() if not g['won'] and g['ante'] == 1)[:int(sys.argv[2])]
T.TW.TIMEOUT = 3600
T.TW.load_lab('/home/claude/balatro-lab')
from balatro.strategy import ValueBot
sim = T.make_sim('/home/claude/game/source')
for s in seeds:
    g = games[s]
    rec, dec = T.play_game(sim, s, ValueBot, False, 1500, False, Replayer(g['acts']))
    # second pass without the teacher: the move texts and the hand at each decision
    sim.reset(s, 'b_red', 1); rows = []
    for d, a in zip(dec, g['acts']):
        o = sim.observe(); st = sim.H.state_name()
        LG = sim.env.lua.globals().G
        hand = ''
        if st == 'SELECTING_HAND':
            cs = LG.hand.cards; hand = ' '.join(f"{cs[i].base.value[:2]}{cs[i].base.suit[0]}" for i in range(1, len(cs)+1))
            js = LG.jokers.cards; hand += ' | ' + ','.join(js[i].ability.name for i in range(1, len(js)+1))
            hand += f' | {LG.GAME.chips}/{LG.GAME.blind.chips} h{LG.GAME.current_round.hands_left} d{LG.GAME.current_round.discards_left}'
        rows.append({'kind': d['kind'], 'student': sim.H.action_desc(a + 1),
                     'teacher': sim.H.action_desc(d['a'] + 1) if d['labeled'] else None, 'hand': hand})
        sim.step(a)
    # the death round: the last run of 'round' decisions
    k = len(rows) - 1
    while k >= 0 and rows[k]['kind'] != 'round': k -= 1
    end = k
    while k >= 0 and rows[k]['kind'] == 'round': k -= 1
    print(json.dumps({'seed': s, 'death_round': rows[k + 1:end + 1]}), flush=True)
