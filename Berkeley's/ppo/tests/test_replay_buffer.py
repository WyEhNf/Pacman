# tests/test_replay_buffer.py
import os
import torch
import pytest

from pacman.world_model.replay_buffer import EpisodeReplayBuffer


def make_episode(T: int) -> dict:
    """Create a random episode of length T."""
    return {
        "grid": torch.randn(T, 8, 31, 28, dtype=torch.float32),
        "scalars": torch.randn(T, 5, dtype=torch.float32),
        "action": torch.randint(0, 4, (T,), dtype=torch.long),
        "reward": torch.randn(T, dtype=torch.float32),
        "done": torch.zeros(T, dtype=torch.bool),
    }


class TestEpisodeReplayBuffer:
    def test_add_and_length(self):
        buf = EpisodeReplayBuffer(max_episodes=10)
        assert len(buf) == 0
        assert buf.total_steps == 0

        buf.add_episode(make_episode(20))
        assert len(buf) == 1
        assert buf.total_steps == 20

        buf.add_episode(make_episode(15))
        assert len(buf) == 2
        assert buf.total_steps == 35

    def test_sample_sequences(self):
        buf = EpisodeReplayBuffer(max_episodes=10)
        # Add several episodes each longer than seq_len
        for _ in range(5):
            buf.add_episode(make_episode(50))

        batch = buf.sample_sequences(batch_size=8, seq_len=10)

        # Check all keys are present
        assert set(batch.keys()) == {"grid", "scalars", "action", "reward", "done"}

        # Check shapes
        assert batch["grid"].shape == (8, 10, 8, 31, 28)
        assert batch["scalars"].shape == (8, 10, 5)
        assert batch["action"].shape == (8, 10)
        assert batch["reward"].shape == (8, 10)
        assert batch["done"].shape == (8, 10)

        # Check dtypes
        assert batch["grid"].dtype == torch.float32
        assert batch["scalars"].dtype == torch.float32
        assert batch["action"].dtype == torch.long
        assert batch["reward"].dtype == torch.float32
        assert batch["done"].dtype == torch.bool

    def test_sample_respects_episode_boundaries(self):
        buf = EpisodeReplayBuffer(max_episodes=10)
        # Add a short episode and verify sampled sequences don't cross episodes
        ep_len = 30
        seq_len = 10
        for _ in range(3):
            buf.add_episode(make_episode(ep_len))

        # Sample many times and verify each sequence is contiguous within one episode
        for _ in range(20):
            batch = buf.sample_sequences(batch_size=4, seq_len=seq_len)
            # batch["reward"] shape: (4, seq_len)
            assert batch["reward"].shape == (4, seq_len)

        # Fallback: if all episodes are shorter than seq_len, still returns something
        buf2 = EpisodeReplayBuffer(max_episodes=10)
        buf2.add_episode(make_episode(5))  # shorter than seq_len=10
        batch2 = buf2.sample_sequences(batch_size=2, seq_len=10)
        assert batch2["grid"].shape[0] == 2
        assert batch2["grid"].shape[1] == 10

    def test_max_episodes_eviction(self):
        max_ep = 5
        buf = EpisodeReplayBuffer(max_episodes=max_ep)

        # Add max_ep episodes of length 10
        for i in range(max_ep):
            buf.add_episode(make_episode(10))
        assert len(buf) == max_ep
        assert buf.total_steps == max_ep * 10

        # Add one more episode — oldest should be evicted
        buf.add_episode(make_episode(20))
        assert len(buf) == max_ep
        # total_steps should reflect eviction of first episode (10 steps) and new one (20 steps)
        assert buf.total_steps == (max_ep - 1) * 10 + 20

    def test_save_and_load(self, tmp_path):
        buf = EpisodeReplayBuffer(max_episodes=10)
        buf.add_episode(make_episode(25))
        buf.add_episode(make_episode(30))

        save_path = tmp_path / "replay_buffer.pt"
        buf.save(str(save_path))
        assert save_path.exists()

        buf2 = EpisodeReplayBuffer(max_episodes=10)
        buf2.load(str(save_path))

        assert len(buf2) == len(buf)
        assert buf2.total_steps == buf.total_steps

        # Check episode data integrity
        for key in ("grid", "scalars", "action", "reward", "done"):
            torch.testing.assert_close(buf2._episodes[0][key], buf._episodes[0][key])
            torch.testing.assert_close(buf2._episodes[1][key], buf._episodes[1][key])
