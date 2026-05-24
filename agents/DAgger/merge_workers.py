"""Merge w1-w4 worker data into single training dataset."""
import sys, os, glob, numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

all_trajs = []
for w in ['w1', 'w2', 'w3', 'w4']:
    files = sorted(glob.glob(os.path.join(PROJECT, 'data', w, 'inc_*.npz')))
    if not files:
        continue
    latest = files[-1]
    d = np.load(latest, allow_pickle=True)
    trajs = list(d['trajectories'])
    print(f'{w}: {len(trajs)} eps from {os.path.basename(latest)}')
    all_trajs.extend(trajs)

scores = [t['score'] for t in all_trajs]
wins = sum(1 for t in all_trajs if t['win'])
print(f'\nTotal: {len(all_trajs)} eps, avg={np.mean(scores):.0f}, wins={wins}/{len(all_trajs)} ({wins/len(all_trajs)*100:.0f}%)')

out = os.path.join(PROJECT, 'data', 'dqn_v3_train.npz')
np.savez_compressed(out, trajectories=np.array(all_trajs, dtype=object))
print(f'Saved: {out}')
