"""Raw PPO training loop — no Trainer, no dashboard, direct control."""
import sys, os, time, numpy as np, torch, yaml, argparse
from pathlib import Path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from pacman.agents.networks import ActorCritic
from pacman.agents.ppo import PPO
from pacman.agents.rollout import RolloutBuffer
from pacman.env.vec_env import VecEnv
from pacman.training.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--total', type=int, default=8000)
parser.add_argument('--run', type=str, default='runs/raw_train')
parser.add_argument('--resume', type=str, default=None)  # checkpoint path
args = parser.parse_args()

# Config
with open('pacman/config/default.yaml') as f: config = yaml.safe_load(f)
config['rewards']['death'] = -50.0
config['training']['checkpoint_every'] = 100
config['training']['eval_every'] = 50

device = torch.device('cuda')
ecfg, ncfg, tcfg, pcfg = config['env'], config['network'], config['training'], config['ppo']
N, T, fs = ecfg['num_envs'], pcfg['rollout_steps'], ecfg.get('frame_stack', 4)

# Model
network = ActorCritic(
    grid_channels=ecfg['observation_channels'] * fs,
    num_scalars=ecfg.get('num_scalar_features', 5),
    cnn_channels=ncfg['cnn_channels'], cnn_kernels=ncfg['cnn_kernels'],
    cnn_strides=ncfg['cnn_strides'], shared_hidden=ncfg['shared_hidden'],
    head_hidden=ncfg['head_hidden'],
).to(device)

ppo = PPO(network, device, pcfg)
rollout = RolloutBuffer(N, T, ecfg['observation_channels'], fs)
vec_env = VecEnv(N, config, difficulty=2)
evaluator = Evaluator(config)

# Reward normalizer (simple running stats)
class RMS:
    def __init__(s, m=0, v=1, c=1e-4): s.m=m; s.v=v; s.c=c
    def update(s, x): bm=x.mean(); bv=x.var(); bc=x.shape[0]; d=bm-s.m; tc=s.c+bc; s.m+=d*bc/tc; s.v=(s.v*s.c+bv*bc+d*d*s.c*bc/tc)/tc; s.c=tc
    def norm(s, x): return np.clip((x-s.m)/(np.sqrt(s.v)+1e-8), -10, 10)

rew_norm = RMS()
start = 0; best_clear = 0.0

# Resume
if args.resume:
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    network.load_state_dict(ckpt['model_state_dict'])
    ppo.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'reward_normalizer' in ckpt:
        ns = ckpt['reward_normalizer']; rew_norm = RMS(ns['mean'], ns['var'], ns['count'])
    start = ckpt['update'] + 1
    best_clear = ckpt.get('clear_rate', 0)
    print(f'Resumed from {args.resume} (update {start})')

os.makedirs(f'{args.run}/checkpoints', exist_ok=True)
print(f'Raw PPO: seed={args.seed} N={N} T={T} total={args.total} start={start}')
ckpt_every = tcfg['checkpoint_every']; eval_every = tcfg['eval_every']
print(f'Checkpoint every {ckpt_every}, eval every {eval_every}\n')

obs = vec_env.reset(seed=args.seed)
t0 = time.time()

for update in range(start, args.total):
    # Anneal LR and entropy
    frac = update / max(args.total, 1)
    ppo.anneal_lr(update, args.total)
    ent_coef = 0.15 + (0.01 - 0.15) * frac ** 0.7
    ppo.set_entropy_coef(ent_coef)

    # Collect rollout
    rollout.reset()
    for _ in range(T):
        masks = vec_env.get_legal_masks()
        actions, log_probs, values = ppo.select_action(obs["grid"], obs["scalars"], masks)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        rollout.add(obs, actions, log_probs, values, rewards, dones)
        obs = next_obs

    # Normalize rewards
    all_r = rollout.rewards.flatten()
    rew_norm.update(all_r[all_r != 0])
    rollout.rewards = rew_norm.norm(rollout.rewards)

    # PPO update
    p_loss, v_loss, ent = ppo.update(rollout, update)

    # Eval
    if update % tcfg['eval_every'] == 0:
        result = evaluator.evaluate(network, 20, device)
        clear = result['level_clear_rate']
        fps = (N * T * tcfg['eval_every']) / max(time.time() - t0, 0.01)
        t0 = time.time()
        ms = result['mean_score']
        print(f'[Upd {update:5d}] clear={clear:.1%} score={ms:.0f} '
              f'p_loss={p_loss:.3f} v_loss={v_loss:.3f} ent={ent:.4f} fps={fps:.0f}')
        if clear > best_clear:
            best_clear = clear
            torch.save({'update':update, 'model_state_dict':network.state_dict(),
                        'optimizer_state_dict':ppo.optimizer.state_dict(),
                        'reward_normalizer':{'mean':rew_norm.m,'var':rew_norm.v,'count':rew_norm.c},
                        'clear_rate':best_clear},
                       f'{args.run}/checkpoints/best.pt')

    # Checkpoint
    if update % tcfg['checkpoint_every'] == 0:
        torch.save({'update':update, 'model_state_dict':network.state_dict(),
                    'optimizer_state_dict':ppo.optimizer.state_dict(),
                    'reward_normalizer':{'mean':rew_norm.m,'var':rew_norm.v,'count':rew_norm.c},
                    'clear_rate':best_clear},
                   f'{args.run}/checkpoints/update_{update}.pt')

print(f'\nDone! Best clear: {best_clear:.1%}')
