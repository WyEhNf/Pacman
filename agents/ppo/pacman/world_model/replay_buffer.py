"""Episode replay buffer for world model training.

Stores complete episodes as dicts of tensors and supports
sampling random contiguous sequences for RSSM training.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List

import torch


Episode = Dict[str, torch.Tensor]

# Expected keys and dtypes for validation
_EPISODE_SCHEMA = {
    "grid": torch.float32,      # (T, 8, 31, 28)
    "scalars": torch.float32,   # (T, 5)
    "action": torch.long,       # (T,)
    "reward": torch.float32,    # (T,)
    "done": torch.bool,         # (T,)
}


class EpisodeReplayBuffer:
    """Stores complete episodes and samples contiguous sub-sequences.

    Args:
        max_episodes: Maximum number of episodes to keep. When the buffer is
            full the oldest episode is evicted before adding a new one.
    """

    def __init__(self, max_episodes: int = 1000) -> None:
        self._max_episodes = max_episodes
        self._episodes: deque[Episode] = deque()
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of frames stored across all episodes."""
        return self._total_steps

    def __len__(self) -> int:
        """Number of episodes currently in the buffer."""
        return len(self._episodes)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        """Add a complete episode to the buffer.

        If the buffer is at capacity the oldest episode is dropped first.

        Args:
            episode: Dict with keys ``grid``, ``scalars``, ``action``,
                ``reward``, and ``done``. The leading dimension of every
                value must match (episode length T).
        """
        T = _episode_length(episode)

        # Evict oldest if at capacity
        if len(self._episodes) >= self._max_episodes:
            evicted = self._episodes.popleft()
            self._total_steps -= _episode_length(evicted)

        self._episodes.append(episode)
        self._total_steps += T

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_sequences(self, batch_size: int, seq_len: int) -> Episode:
        """Sample a batch of contiguous sub-sequences from stored episodes.

        Preferentially samples from episodes that are at least ``seq_len``
        frames long. If no such episodes exist, falls back to all episodes
        and pads / wraps the sequence by cycling within the episode.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Length of each sequence.

        Returns:
            Dict of stacked tensors with shape ``(batch_size, seq_len, ...)``.
        """
        if len(self._episodes) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        # Prefer episodes that are long enough
        long_episodes: List[Episode] = [
            ep for ep in self._episodes if _episode_length(ep) >= seq_len
        ]
        pool = long_episodes if long_episodes else list(self._episodes)

        sequences: List[Episode] = []
        for _ in range(batch_size):
            ep = random.choice(pool)
            T = _episode_length(ep)

            if T >= seq_len:
                start = random.randint(0, T - seq_len)
                seq = {k: v[start : start + seq_len] for k, v in ep.items()}
            else:
                # Episode is shorter than seq_len: tile and truncate
                repeats = (seq_len + T - 1) // T  # ceiling division
                seq = {
                    k: v.repeat(
                        (repeats,) + (1,) * (v.dim() - 1)
                    )[:seq_len]
                    for k, v in ep.items()
                }

            sequences.append(seq)

        # Stack into (batch_size, seq_len, ...)
        return {
            key: torch.stack([s[key] for s in sequences], dim=0)
            for key in sequences[0]
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the buffer to disk using ``torch.save``.

        Args:
            path: File path to write the buffer state to.
        """
        state = {
            "max_episodes": self._max_episodes,
            "total_steps": self._total_steps,
            "episodes": list(self._episodes),
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Restore the buffer from a file written by :meth:`save`.

        Replaces any existing content in the buffer.

        Args:
            path: File path to read the buffer state from.
        """
        state = torch.load(path, weights_only=False)
        self._max_episodes = state["max_episodes"]
        self._total_steps = state["total_steps"]
        self._episodes = deque(state["episodes"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _episode_length(episode: Episode) -> int:
    """Return the number of time steps in an episode."""
    # All tensors share the same leading dimension; grab from first key.
    key = next(iter(episode))
    return episode[key].shape[0]
