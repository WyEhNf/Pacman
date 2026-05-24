"""
Trajectory Dataset for Decision Transformer training.

Loads the .npz produced by scripts/collect_expert_data.py and emits
fixed-length windows of (returns_to_go, state, action) triplets.

Handles:
  - Variable trajectory lengths  → random window sampling
  - Variable state dimensions    → padding to a fixed max
  - Quality weights              → weighted sampling
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Loads expert trajectories and yields (RtG, state, action) windows.

    Usage:
        ds = TrajectoryDataset('data/expert_trajectories.npz',
                                context_len=20, state_dim=448)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for rtg, states, actions, masks in loader:
            ...
    """

    def __init__(self, npz_path, context_len=20, state_dim=None,
                 min_rtg=None, max_rtg=None):
        """
        Args:
            npz_path:     path to the .npz file from collect_expert_data.py
            context_len:  K, number of consecutive steps per window
            state_dim:    pad/truncate state vectors to this size.
                          If None, auto-detect from data.
            min_rtg:      normalise RtG: shift lower bound (auto if None)
            max_rtg:      normalise RtG: scale upper bound (auto if None)
        """
        data = np.load(npz_path, allow_pickle=True)
        self.trajectories = data['trajectories']  # list of dicts

        self.context_len = context_len

        # ── Determine state dimension ──
        if state_dim is None:
            state_dim = int(max(
                traj['states'].shape[-1] for traj in self.trajectories))
        self.state_dim = state_dim

        # ── Determine RtG normalisation range ──
        all_rtg = np.concatenate([t['returns_to_go'] for t in self.trajectories])
        self.rtg_min = float(np.min(all_rtg)) if min_rtg is None else min_rtg
        self.rtg_max = float(np.max(all_rtg)) if max_rtg is None else max_rtg
        self.rtg_scale = max(self.rtg_max - self.rtg_min, 1.0)

        # ── Build index: (traj_idx, start_step) for every valid window ──
        self.windows = []
        self.weights = []  # quality_weight for each window

        for i, traj in enumerate(self.trajectories):
            T = len(traj['actions'])
            if T < context_len:
                # Pad short trajectories
                self.windows.append((i, 0))
                self.weights.append(traj.get('quality_weight', 1.0))
            else:
                # Sample any valid start position; add one at start and one at end
                # so we can always get the initial and final contexts.
                for start in range(0, T - context_len + 1, context_len // 2):
                    self.windows.append((i, start))
                    self.weights.append(traj.get('quality_weight', 1.0))
                # Always include the tail window
                if T - context_len >= 0:
                    last_start = T - context_len
                    if (i, last_start) not in self.windows:
                        self.windows.append((i, last_start))
                        self.weights.append(traj.get('quality_weight', 1.0))

        if len(self.windows) == 0:
            raise RuntimeError("No trajectories long enough for context_len. "
                               f"Got {len(self.trajectories)} trajectories.")

        self._weights = np.array(self.weights, dtype=np.float32)
        self._weights /= self._weights.sum()

        print(f"TrajectoryDataset: {len(self.trajectories)} trajectories, "
              f"{len(self.windows)} windows, state_dim={self.state_dim}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        traj_idx, start = self.windows[idx]
        traj = self.trajectories[traj_idx]
        T = len(traj['actions'])
        end = min(start + self.context_len, T)

        # ── Extract slice ──
        rtg    = traj['returns_to_go'][start:end].astype(np.float32)
        states = traj['states'][start:end].astype(np.float32)
        acts   = traj['actions'][start:end].astype(np.int64)

        # ── Normalise RtG ──
        rtg = (rtg - self.rtg_min) / self.rtg_scale

        # ── Pad state to fixed dimension ──
        if states.shape[-1] < self.state_dim:
            pad = np.zeros((states.shape[0], self.state_dim - states.shape[-1]),
                           dtype=np.float32)
            states = np.concatenate([states, pad], axis=-1)
        elif states.shape[-1] > self.state_dim:
            states = states[:, :self.state_dim]

        # ── Pad sequence to context_len ──
        pad_len = self.context_len - (end - start)
        if pad_len > 0:
            rtg    = np.pad(rtg,    ((0, pad_len),), constant_values=0)
            states = np.pad(states, ((0, pad_len), (0, 0)), constant_values=0)
            acts   = np.pad(acts,   ((0, pad_len),), constant_values=0)
            # attention mask: 1 = real, 0 = pad
            mask = np.concatenate([
                np.ones(end - start, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32),
            ])
        else:
            mask = np.ones(self.context_len, dtype=np.float32)

        return (
            torch.from_numpy(rtg).unsqueeze(-1),    # (K, 1)
            torch.from_numpy(states),                # (K, state_dim)
            torch.from_numpy(acts),                  # (K,)
            torch.from_numpy(mask),                  # (K,)
        )

    def get_rtg_stats(self):
        """Return (min, max, scale) for reconstructing RtG in inference."""
        return self.rtg_min, self.rtg_max, self.rtg_scale


# ─── Quick test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/expert_trajectories.npz'
    ds = TrajectoryDataset(path, context_len=20)
    rtg, states, actions, mask = ds[0]
    print(f"Sample: rtg={rtg.shape}, states={states.shape}, "
          f"actions={actions.shape}, mask={mask.shape}")
    print(f"RtG range: [{ds.rtg_min:.1f}, {ds.rtg_max:.1f}]")
    print(f"Quality weight sum: {ds._weights.sum():.3f}")
