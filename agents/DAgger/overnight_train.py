"""Overnight automated training pipeline."""
import sys, os, time, subprocess
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = r"C:\Users\admin\miniconda3\envs\pacman\python.exe"
DATA_DIR = os.path.join(PROJECT, 'data')
CKPT_DIR = os.path.join(PROJECT, 'checkpoints')
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

log = open(os.path.join(PROJECT, 'overnight_log.txt'), 'w')

def run(cmd, desc):
    print(f'\n{"="*60}\n{desc}\n{"="*60}')
    log.write(f'\n{desc}\n')
    log.flush()
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=PROJECT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', errors='replace')
    print(out[-2000:])  # last 2000 chars
    log.write(out)
    log.flush()
    print(f'[{desc}] done in {time.time()-t0:.0f}s, exit={r.returncode}')
    return r.returncode

# Phase 1: Collect 300 episodes with betterEvaluationFunction
rc = run(f'{PYTHON} -u scripts/run_collection.py --episodes 300 --output data/better_eval',
         'Phase 1: Collect 300 expert episodes (betterEvaluationFunction)')
if rc != 0:
    print('Collection failed, aborting')
    log.close()
    sys.exit(1)

# Find latest inc file
import glob
incs = sorted(glob.glob(os.path.join(DATA_DIR, 'better_eval', 'inc_*.npz')))
data_file = incs[-1] if incs else os.path.join(DATA_DIR, 'better_eval', 'expert_trajectories.npz')
print(f'Training on: {data_file}')

# Phase 2: Train DT_v3
rc = run(f'{PYTHON} -u scripts/train_bc_quick.py {data_file}',
         'Phase 2: Train DT_v3 (BC)')
if rc != 0:
    print('Training failed')
    log.close()
    sys.exit(1)

# Copy checkpoint
import shutil
src = os.path.join(CKPT_DIR, 'dt_v1_100ep.pt')
dst = os.path.join(CKPT_DIR, 'dt_v3_better_eval.pt')
shutil.copy(src, dst)
print(f'Checkpoint: {dst}')

# Phase 3: Evaluate
run(f'{PYTHON} -u scripts/test_dt.py',
    'Phase 3: Quick evaluation')

print('\nOvernight training complete.')
log.write('\nDone.\n'); log.close()
