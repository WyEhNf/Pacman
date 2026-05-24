# Dreaming Pac-Man Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an RSSM world model that learns to simulate Pac-Man from gameplay data, then train an RL agent entirely inside the model's imagination.

**Architecture:** Three-phase pipeline: (A) collect gameplay data from trained PPO agent, (B) train an RSSM world model on that data, (C) train a new PPO agent purely in the world model's imagination. Visualization shows real game vs. dreamed game side-by-side.

**Tech Stack:** PyTorch (MPS backend), NumPy, pygame, existing Pac-Man engine

---

### Task 1: Replay Buffer

**Files:**
- Create: `pacman/world_model/__init__.py`
- Create: `pacman/world_model/replay_buffer.py`
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests for replay buffer**

```python
# tests/test_replay_buffer.py
import torch
import pytest
from pacman.world_model.replay_buffer import EpisodeReplayBuffer


class TestEpisodeReplayBuffer:
    def test_add_and_length(self):
        buf = EpisodeReplayBuffer(max_episodes=100)
        episode = {
            "grid": torch.randn(50, 8, 31, 28),
            "scalars": torch.randn(50, 5),
            "action": torch.randint(0, 4, (50,)),
            "reward": torch.randn(50),
            "done": torch.zeros(50, dtype=torch.bool),
        }
        episode["done"][-1] = True
        buf.add_episode(episode)
        assert len(buf) == 1
        assert buf.total_steps == 50

    def test_sample_sequences(self):
        buf = EpisodeReplayBuffer(max_episodes=100)
        for _ in range(10):
            ep_len = 100
            episode = {
                "grid": torch.randn(ep_len, 8, 31, 28),
                "scalars": torch.randn(ep_len, 5),
                "action": torch.randint(0, 4, (ep_len,)),
                "reward": torch.randn(ep_len),
                "done": torch.zeros(ep_len, dtype=torch.bool),
            }
            episode["done"][-1] = True
            buf.add_episode(episode)

        batch = buf.sample_sequences(batch_size=4, seq_len=20)
        assert batch["grid"].shape == (4, 20, 8, 31, 28)
        assert batch["scalars"].shape == (4, 20, 5)
        assert batch["action"].shape == (4, 20)
        assert batch["reward"].shape == (4, 20)
        assert batch["done"].shape == (4, 20)

    def test_sample_respects_episode_boundaries(self):
        buf = EpisodeReplayBuffer(max_episodes=100)
        # Add short episode
        episode = {
            "grid": torch.randn(10, 8, 31, 28),
            "scalars": torch.randn(10, 5),
            "action": torch.randint(0, 4, (10,)),
            "reward": torch.randn(10),
            "done": torch.zeros(10, dtype=torch.bool),
        }
        episode["done"][-1] = True
        buf.add_episode(episode)

        # seq_len > episode length should still work (pad or pick longer episodes)
        # With only one short episode, sampling seq_len=10 should work
        batch = buf.sample_sequences(batch_size=1, seq_len=10)
        assert batch["grid"].shape == (1, 10, 8, 31, 28)

    def test_max_episodes_eviction(self):
        buf = EpisodeReplayBuffer(max_episodes=5)
        for i in range(10):
            episode = {
                "grid": torch.randn(20, 8, 31, 28),
                "scalars": torch.randn(20, 5),
                "action": torch.randint(0, 4, (20,)),
                "reward": torch.randn(20),
                "done": torch.zeros(20, dtype=torch.bool),
            }
            episode["done"][-1] = True
            buf.add_episode(episode)
        assert len(buf) == 5

    def test_save_and_load(self, tmp_path):
        buf = EpisodeReplayBuffer(max_episodes=100)
        episode = {
            "grid": torch.randn(30, 8, 31, 28),
            "scalars": torch.randn(30, 5),
            "action": torch.randint(0, 4, (30,)),
            "reward": torch.randn(30),
            "done": torch.zeros(30, dtype=torch.bool),
        }
        episode["done"][-1] = True
        buf.add_episode(episode)
        buf.save(tmp_path / "buffer.pt")

        buf2 = EpisodeReplayBuffer.load(tmp_path / "buffer.pt")
        assert len(buf2) == 1
        assert buf2.total_steps == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_replay_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pacman.world_model'`

- [ ] **Step 3: Implement replay buffer**

```python
# pacman/world_model/__init__.py
```

```python
# pacman/world_model/replay_buffer.py
"""Sequential episode replay buffer for world model training."""
from pathlib import Path
import torch


class EpisodeReplayBuffer:
    """Stores complete episodes for sequential sampling.

    Each episode is a dict of tensors keyed by:
      grid:    (T, 8, 31, 28)
      scalars: (T, 5)
      action:  (T,)
      reward:  (T,)
      done:    (T,) bool
    """

    def __init__(self, max_episodes: int = 10000):
        self.max_episodes = max_episodes
        self.episodes: list[dict[str, torch.Tensor]] = []
        self.total_steps = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: dict[str, torch.Tensor]) -> None:
        ep_len = episode["grid"].shape[0]
        if len(self.episodes) >= self.max_episodes:
            removed = self.episodes.pop(0)
            self.total_steps -= removed["grid"].shape[0]
        self.episodes.append(episode)
        self.total_steps += ep_len

    def sample_sequences(
        self, batch_size: int, seq_len: int,
    ) -> dict[str, torch.Tensor]:
        """Sample random contiguous sequences from stored episodes."""
        grids, scalars, actions, rewards, dones = [], [], [], [], []
        # Filter episodes that are at least seq_len long
        valid = [ep for ep in self.episodes if ep["grid"].shape[0] >= seq_len]
        if not valid:
            valid = self.episodes  # fallback: use all episodes

        indices = torch.randint(0, len(valid), (batch_size,))
        for idx in indices:
            ep = valid[idx.item()]
            ep_len = ep["grid"].shape[0]
            max_start = max(0, ep_len - seq_len)
            start = torch.randint(0, max_start + 1, (1,)).item()
            end = start + seq_len
            grids.append(ep["grid"][start:end])
            scalars.append(ep["scalars"][start:end])
            actions.append(ep["action"][start:end])
            rewards.append(ep["reward"][start:end])
            dones.append(ep["done"][start:end])

        return {
            "grid": torch.stack(grids),
            "scalars": torch.stack(scalars),
            "action": torch.stack(actions),
            "reward": torch.stack(rewards),
            "done": torch.stack(dones),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"episodes": self.episodes, "total_steps": self.total_steps}, path)

    @classmethod
    def load(cls, path: Path) -> "EpisodeReplayBuffer":
        data = torch.load(path, map_location="cpu", weights_only=False)
        buf = cls()
        buf.episodes = data["episodes"]
        buf.total_steps = data["total_steps"]
        return buf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_replay_buffer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pacman/world_model/__init__.py pacman/world_model/replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat(world-model): add episode replay buffer with sequential sampling"
```

---

### Task 2: RSSM Core — Encoder, Decoder, Heads

**Files:**
- Create: `pacman/world_model/encoder.py`
- Create: `pacman/world_model/decoder.py`
- Create: `pacman/world_model/heads.py`
- Test: `tests/test_world_model_components.py`

- [ ] **Step 1: Write failing tests for encoder, decoder, and heads**

```python
# tests/test_world_model_components.py
import torch
import pytest
from pacman.world_model.encoder import ObservationEncoder
from pacman.world_model.decoder import ObservationDecoder
from pacman.world_model.heads import RewardHead, ContinueHead


class TestObservationEncoder:
    def test_output_shape(self):
        enc = ObservationEncoder(
            grid_channels=8, num_scalars=5,
            cnn_channels=[64, 128, 256, 256],
            cnn_kernels=[4, 4, 4, 4],
            cnn_strides=[2, 2, 2, 1],
            output_dim=512,
        )
        grid = torch.randn(4, 8, 31, 28)
        scalars = torch.randn(4, 5)
        out = enc(grid, scalars)
        assert out.shape == (4, 512)

    def test_gradients_flow(self):
        enc = ObservationEncoder(
            grid_channels=8, num_scalars=5,
            cnn_channels=[64, 128, 256, 256],
            cnn_kernels=[4, 4, 4, 4],
            cnn_strides=[2, 2, 2, 1],
            output_dim=512,
        )
        grid = torch.randn(2, 8, 31, 28)
        scalars = torch.randn(2, 5)
        out = enc(grid, scalars)
        out.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None


class TestObservationDecoder:
    def test_output_shape(self):
        dec = ObservationDecoder(
            latent_dim=2560, grid_channels=8, num_scalars=5,
            cnn_channels=[256, 256, 128, 64],
            cnn_kernels=[4, 4, 4, 4],
            cnn_strides=[1, 2, 2, 2],
        )
        latent = torch.randn(4, 2560)
        grid_out, scalars_out = dec(latent)
        assert grid_out.shape == (4, 8, 31, 28)
        assert scalars_out.shape == (4, 5)

    def test_gradients_flow(self):
        dec = ObservationDecoder(
            latent_dim=2560, grid_channels=8, num_scalars=5,
            cnn_channels=[256, 256, 128, 64],
            cnn_kernels=[4, 4, 4, 4],
            cnn_strides=[1, 2, 2, 2],
        )
        latent = torch.randn(2, 2560)
        grid_out, scalars_out = dec(latent)
        (grid_out.sum() + scalars_out.sum()).backward()
        for p in dec.parameters():
            assert p.grad is not None


class TestHeads:
    def test_reward_head_shape(self):
        head = RewardHead(latent_dim=2560, hidden_dim=512, num_layers=3)
        latent = torch.randn(4, 2560)
        out = head(latent)
        assert out.shape == (4, 1)

    def test_continue_head_shape(self):
        head = ContinueHead(latent_dim=2560, hidden_dim=512, num_layers=3)
        latent = torch.randn(4, 2560)
        out = head(latent)
        assert out.shape == (4, 1)

    def test_continue_head_outputs_logits(self):
        head = ContinueHead(latent_dim=2560, hidden_dim=512, num_layers=3)
        latent = torch.randn(4, 2560)
        out = head(latent)
        # Should output raw logits (not sigmoid), so values can be any range
        assert out.min() < 0 or out.max() > 1  # at least one outside [0,1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_world_model_components.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the encoder**

```python
# pacman/world_model/encoder.py
"""CNN encoder: observation -> latent representation."""
import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encodes (grid, scalars) observation into a flat latent vector."""

    def __init__(
        self,
        grid_channels: int = 8,
        num_scalars: int = 5,
        cnn_channels: list[int] = (64, 128, 256, 256),
        cnn_kernels: list[int] = (4, 4, 4, 4),
        cnn_strides: list[int] = (2, 2, 2, 1),
        output_dim: int = 512,
    ):
        super().__init__()
        layers = []
        in_ch = grid_channels
        for out_ch, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2))
            layers.append(nn.SiLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, 31, 28)
            cnn_flat = self.cnn(dummy).view(1, -1).shape[1]

        self.scalar_embed = nn.Sequential(
            nn.Linear(num_scalars, 64),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_flat + 64, output_dim),
            nn.SiLU(),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(grid).flatten(1)
        scalar_out = self.scalar_embed(scalars)
        combined = torch.cat([cnn_out, scalar_out], dim=-1)
        return self.fc(combined)
```

- [ ] **Step 4: Implement the decoder**

```python
# pacman/world_model/decoder.py
"""CNN decoder: latent representation -> reconstructed observation."""
import torch
import torch.nn as nn


class ObservationDecoder(nn.Module):
    """Decodes latent vector back to (grid, scalars) observation."""

    def __init__(
        self,
        latent_dim: int = 2560,
        grid_channels: int = 8,
        num_scalars: int = 5,
        cnn_channels: list[int] = (256, 256, 128, 64),
        cnn_kernels: list[int] = (4, 4, 4, 4),
        cnn_strides: list[int] = (1, 2, 2, 2),
    ):
        super().__init__()
        # Compute the spatial size we need to start the transposed CNN
        # After encoding (31,28) with strides [2,2,2,1]: roughly (4,4)
        # We'll use a fixed start size and let the network learn to upsample
        self._start_channels = cnn_channels[0]
        self._start_h = 4
        self._start_w = 4
        start_flat = self._start_channels * self._start_h * self._start_w

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, start_flat),
            nn.SiLU(),
        )

        layers = []
        in_ch = cnn_channels[0]
        for i, (out_ch, k, s) in enumerate(zip(cnn_channels[1:], cnn_kernels[:-1], cnn_strides[:-1])):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, k, stride=s, padding=k // 2, output_padding=s - 1))
            layers.append(nn.SiLU())
            in_ch = out_ch
        # Final layer: output grid_channels, no activation
        layers.append(nn.ConvTranspose2d(in_ch, grid_channels, cnn_kernels[-1], stride=cnn_strides[-1],
                                          padding=cnn_kernels[-1] // 2, output_padding=cnn_strides[-1] - 1))
        self.deconv = nn.Sequential(*layers)

        # Adaptive layer to match exact output size (31, 28)
        self.grid_adapt = nn.AdaptiveAvg2d((31, 28))  # compat layer

        # Scalar reconstruction
        self.scalar_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_scalars),
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = latent.shape[0]
        x = self.fc(latent)
        x = x.view(batch, self._start_channels, self._start_h, self._start_w)
        x = self.deconv(x)
        grid = self.grid_adapt(x)  # ensure exact (31, 28)
        scalars = self.scalar_head(latent)
        return grid, scalars


class AdaptiveAvg2d(nn.Module):
    """Simple wrapper for adaptive average pooling to target size."""
    def __init__(self, target_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        return self.pool(x)
```

Note: The decoder uses `nn.AdaptiveAvgPool2d` as a final layer to guarantee exact (31, 28) output. In `__init__`, `self.grid_adapt` should be `nn.AdaptiveAvgPool2d((31, 28))` (fix the typo `AdaptiveAvg2d` to `AdaptiveAvgPool2d`).

- [ ] **Step 5: Implement the heads**

```python
# pacman/world_model/heads.py
"""Reward and continue prediction heads."""
import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    """Build an MLP with SiLU activations."""
    layers = []
    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class RewardHead(nn.Module):
    """Predicts reward from latent state."""

    def __init__(self, latent_dim: int = 2560, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.mlp = build_mlp(latent_dim, hidden_dim, 1, num_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent)


class ContinueHead(nn.Module):
    """Predicts continuation probability (logits) from latent state."""

    def __init__(self, latent_dim: int = 2560, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.mlp = build_mlp(latent_dim, hidden_dim, 1, num_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent)  # raw logits; apply sigmoid externally
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_world_model_components.py -v`
Expected: All 7 tests PASS

- [ ] **Step 7: Commit**

```bash
git add pacman/world_model/encoder.py pacman/world_model/decoder.py pacman/world_model/heads.py tests/test_world_model_components.py
git commit -m "feat(world-model): add encoder, decoder, reward and continue heads"
```

---

### Task 3: RSSM Dynamics Model

**Files:**
- Create: `pacman/world_model/rssm.py`
- Test: `tests/test_rssm.py`

- [ ] **Step 1: Write failing tests for RSSM**

```python
# tests/test_rssm.py
import torch
import pytest
from pacman.world_model.rssm import RSSM


@pytest.fixture
def rssm():
    return RSSM(
        stoch_classes=32, stoch_categoricals=64,
        gru_hidden=512, action_dim=4,
        encoder_output_dim=512,
    )


class TestRSSM:
    def test_initial_state_shape(self, rssm):
        h, z = rssm.initial_state(batch_size=4)
        assert h.shape == (4, 512)   # GRU hidden
        assert z.shape == (4, 2048)  # 32 * 64

    def test_dynamics_step(self, rssm):
        h, z = rssm.initial_state(batch_size=4)
        action = torch.randint(0, 4, (4,))
        h_next = rssm.dynamics(h, z, action)
        assert h_next.shape == (4, 512)

    def test_prior(self, rssm):
        h = torch.randn(4, 512)
        z_prior, prior_logits = rssm.prior(h)
        assert z_prior.shape == (4, 2048)  # 32 * 64 flattened
        assert prior_logits.shape == (4, 32, 64)

    def test_posterior(self, rssm):
        h = torch.randn(4, 512)
        encoder_out = torch.randn(4, 512)
        z_post, post_logits = rssm.posterior(h, encoder_out)
        assert z_post.shape == (4, 2048)
        assert post_logits.shape == (4, 32, 64)

    def test_latent_dim_property(self, rssm):
        assert rssm.latent_dim == 2560  # 512 + 2048

    def test_full_forward_sequence(self, rssm):
        """Test a 5-step forward pass with observations."""
        batch = 4
        seq_len = 5
        h, z = rssm.initial_state(batch_size=batch)
        encoder_outputs = torch.randn(batch, seq_len, 512)
        actions = torch.randint(0, 4, (batch, seq_len))

        all_h, all_z_prior, all_z_post = [], [], []
        all_prior_logits, all_post_logits = [], []

        for t in range(seq_len):
            h = rssm.dynamics(h, z, actions[:, t])
            z_prior, prior_logits = rssm.prior(h)
            z_post, post_logits = rssm.posterior(h, encoder_outputs[:, t])
            z = z_post  # use posterior during training
            all_h.append(h)
            all_z_post.append(z_post)
            all_prior_logits.append(prior_logits)
            all_post_logits.append(post_logits)

        h_seq = torch.stack(all_h, dim=1)
        assert h_seq.shape == (batch, seq_len, 512)

    def test_gradients_flow(self, rssm):
        h, z = rssm.initial_state(batch_size=2)
        action = torch.randint(0, 4, (2,))
        encoder_out = torch.randn(2, 512)

        h = rssm.dynamics(h, z, action)
        z_post, _ = rssm.posterior(h, encoder_out)
        loss = z_post.sum() + h.sum()
        loss.backward()
        for name, p in rssm.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rssm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RSSM**

```python
# pacman/world_model/rssm.py
"""Recurrent State-Space Model (RSSM) for world model dynamics."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """RSSM with categorical stochastic state and GRU deterministic state.

    Latent state = (h, z) where:
      h: deterministic GRU hidden (gru_hidden,)
      z: stochastic categorical, flattened (stoch_classes * stoch_categoricals,)
    """

    def __init__(
        self,
        stoch_classes: int = 32,
        stoch_categoricals: int = 64,
        gru_hidden: int = 512,
        action_dim: int = 4,
        encoder_output_dim: int = 512,
    ):
        super().__init__()
        self.stoch_classes = stoch_classes
        self.stoch_categoricals = stoch_categoricals
        self.stoch_dim = stoch_classes * stoch_categoricals
        self.gru_hidden = gru_hidden
        self.action_dim = action_dim

        # Action embedding
        self.action_embed = nn.Embedding(action_dim, 64)

        # GRU input: z + action_embed
        gru_input_dim = self.stoch_dim + 64
        self.gru_pre = nn.Sequential(
            nn.Linear(gru_input_dim, gru_hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(gru_hidden, gru_hidden)

        # Prior: h -> z_prior logits
        self.prior_net = nn.Sequential(
            nn.Linear(gru_hidden, 512),
            nn.SiLU(),
            nn.Linear(512, stoch_classes * stoch_categoricals),
        )

        # Posterior: (h, encoder_out) -> z_posterior logits
        self.posterior_net = nn.Sequential(
            nn.Linear(gru_hidden + encoder_output_dim, 512),
            nn.SiLU(),
            nn.Linear(512, stoch_classes * stoch_categoricals),
        )

    @property
    def latent_dim(self) -> int:
        return self.gru_hidden + self.stoch_dim

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h, z)."""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.gru_hidden, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def dynamics(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Advance deterministic state: (h_t, z_t, a_t) -> h_{t+1}."""
        action_emb = self.action_embed(action)
        gru_in = self.gru_pre(torch.cat([z, action_emb], dim=-1))
        h_next = self.gru(gru_in, h)
        return h_next

    def prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior z from deterministic state only."""
        logits = self.prior_net(h)
        logits = logits.view(-1, self.stoch_classes, self.stoch_categoricals)
        z = self._sample_categorical(logits)
        return z, logits

    def posterior(
        self, h: torch.Tensor, encoder_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior z from deterministic state + encoder output."""
        logits = self.posterior_net(torch.cat([h, encoder_out], dim=-1))
        logits = logits.view(-1, self.stoch_classes, self.stoch_categoricals)
        z = self._sample_categorical(logits)
        return z, logits

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradients."""
        # logits: (batch, stoch_classes, stoch_categoricals)
        probs = F.softmax(logits, dim=-1)
        # Straight-through: sample onehot, but use softmax for gradients
        if self.training:
            # Gumbel-softmax straight-through
            onehot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            # Argmax at eval time
            indices = logits.argmax(dim=-1)
            onehot = F.one_hot(indices, self.stoch_categoricals).float()
        # Flatten: (batch, stoch_classes * stoch_categoricals)
        return onehot.view(-1, self.stoch_dim)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rssm.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pacman/world_model/rssm.py tests/test_rssm.py
git commit -m "feat(world-model): add RSSM dynamics with categorical stochastic state"
```

---

### Task 4: WorldModel Integration Class

**Files:**
- Create: `pacman/world_model/world_model.py`
- Test: `tests/test_world_model.py`

- [ ] **Step 1: Write failing tests for the integrated world model**

```python
# tests/test_world_model.py
import torch
import pytest
from pacman.world_model.world_model import WorldModel


@pytest.fixture
def wm():
    return WorldModel()


class TestWorldModel:
    def test_train_step_returns_losses(self, wm):
        batch = {
            "grid": torch.randn(4, 20, 8, 31, 28),
            "scalars": torch.randn(4, 20, 5),
            "action": torch.randint(0, 4, (4, 20)),
            "reward": torch.randn(4, 20),
            "done": torch.zeros(4, 20, dtype=torch.bool),
        }
        losses = wm.train_step(batch)
        assert "reconstruction" in losses
        assert "reward" in losses
        assert "continue" in losses
        assert "kl" in losses
        assert "total" in losses
        for v in losses.values():
            assert torch.isfinite(torch.tensor(v)), f"Non-finite loss: {v}"

    def test_imagine_shapes(self, wm):
        wm.eval()
        # Start from a real observation
        grid = torch.randn(8, 8, 31, 28)
        scalars = torch.randn(8, 5)
        action_fn = lambda h, z: torch.randint(0, 4, (h.shape[0],))
        imagined = wm.imagine(grid, scalars, action_fn, horizon=10)
        assert imagined["h"].shape == (8, 10, 512)
        assert imagined["z"].shape == (8, 10, 2048)
        assert imagined["reward"].shape == (8, 10)
        assert imagined["cont"].shape == (8, 10)

    def test_decode(self, wm):
        wm.eval()
        h = torch.randn(4, 512)
        z = torch.randn(4, 2048)
        grid, scalars = wm.decode(h, z)
        assert grid.shape == (4, 8, 31, 28)
        assert scalars.shape == (4, 5)

    def test_parameter_count(self, wm):
        total = sum(p.numel() for p in wm.parameters())
        assert 5_000_000 < total < 15_000_000  # ~8-10M expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_world_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement WorldModel**

```python
# pacman/world_model/world_model.py
"""Integrated world model: RSSM + encoder + decoder + heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rssm import RSSM
from .encoder import ObservationEncoder
from .decoder import ObservationDecoder
from .heads import RewardHead, ContinueHead


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class WorldModel(nn.Module):
    """Complete world model: encode observations, predict dynamics, decode."""

    def __init__(
        self,
        grid_channels: int = 8,
        num_scalars: int = 5,
        stoch_classes: int = 32,
        stoch_categoricals: int = 64,
        gru_hidden: int = 512,
        action_dim: int = 4,
        free_nats: float = 1.0,
        kl_beta: float = 1.0,
    ):
        super().__init__()
        self.free_nats = free_nats
        self.kl_beta = kl_beta

        self.encoder = ObservationEncoder(
            grid_channels=grid_channels, num_scalars=num_scalars,
            output_dim=512,
        )
        self.rssm = RSSM(
            stoch_classes=stoch_classes,
            stoch_categoricals=stoch_categoricals,
            gru_hidden=gru_hidden,
            action_dim=action_dim,
            encoder_output_dim=512,
        )
        latent_dim = self.rssm.latent_dim  # 2560

        self.decoder = ObservationDecoder(
            latent_dim=latent_dim,
            grid_channels=grid_channels,
            num_scalars=num_scalars,
        )
        self.reward_head = RewardHead(latent_dim=latent_dim)
        self.continue_head = ContinueHead(latent_dim=latent_dim)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One training step on a batch of sequences.

        Args:
            batch: dict with keys grid (B,T,C,H,W), scalars (B,T,S),
                   action (B,T), reward (B,T), done (B,T)

        Returns: dict of scalar losses
        """
        B, T = batch["action"].shape
        device = batch["grid"].device

        # Encode all observations
        grid_flat = batch["grid"].reshape(B * T, *batch["grid"].shape[2:])
        scalars_flat = batch["scalars"].reshape(B * T, *batch["scalars"].shape[2:])
        encoder_out = self.encoder(grid_flat, scalars_flat)
        encoder_out = encoder_out.view(B, T, -1)

        # Roll out RSSM
        h, z = self.rssm.initial_state(B)
        h, z = h.to(device), z.to(device)

        all_h, all_z = [], []
        all_prior_logits, all_post_logits = [], []

        for t in range(T):
            h = self.rssm.dynamics(h, z, batch["action"][:, t])
            z_prior, prior_logits = self.rssm.prior(h)
            z_post, post_logits = self.rssm.posterior(h, encoder_out[:, t])
            z = z_post  # use posterior during training

            all_h.append(h)
            all_z.append(z)
            all_prior_logits.append(prior_logits)
            all_post_logits.append(post_logits)

        # Stack: (B, T, dim)
        h_seq = torch.stack(all_h, dim=1)
        z_seq = torch.stack(all_z, dim=1)
        prior_logits = torch.stack(all_prior_logits, dim=1)
        post_logits = torch.stack(all_post_logits, dim=1)

        # Concatenate latent for decoder/heads
        latent = torch.cat([h_seq, z_seq], dim=-1)  # (B, T, 2560)
        latent_flat = latent.reshape(B * T, -1)

        # Decode
        grid_pred, scalars_pred = self.decoder(latent_flat)
        grid_pred = grid_pred.view(B, T, *grid_pred.shape[1:])
        scalars_pred = scalars_pred.view(B, T, -1)

        # Reward and continue predictions
        reward_pred = self.reward_head(latent_flat).view(B, T)
        continue_pred = self.continue_head(latent_flat).view(B, T)

        # --- Losses ---
        # Reconstruction
        recon_loss = F.mse_loss(grid_pred, batch["grid"]) + F.mse_loss(scalars_pred, batch["scalars"])

        # Reward (symlog transformed)
        reward_target = symlog(batch["reward"])
        reward_loss = F.mse_loss(reward_pred, reward_target)

        # Continue
        continue_target = (~batch["done"]).float()
        continue_loss = F.binary_cross_entropy_with_logits(continue_pred, continue_target)

        # KL divergence between posterior and prior (categorical)
        prior_probs = F.softmax(prior_logits, dim=-1)
        post_probs = F.softmax(post_logits, dim=-1)
        kl = torch.sum(
            post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8)),
            dim=-1,
        ).mean()
        kl = torch.clamp(kl - self.free_nats, min=0.0)

        total = recon_loss + reward_loss + continue_loss + self.kl_beta * kl

        return {
            "reconstruction": recon_loss.item(),
            "reward": reward_loss.item(),
            "continue": continue_loss.item(),
            "kl": kl.item(),
            "total": total.item(),
            "_total_tensor": total,  # for backward
        }

    @torch.no_grad()
    def imagine(
        self,
        start_grid: torch.Tensor,    # (B, C, H, W)
        start_scalars: torch.Tensor,  # (B, S)
        action_fn,                     # callable(h, z) -> actions (B,)
        horizon: int = 15,
    ) -> dict[str, torch.Tensor]:
        """Roll out imagination from a starting observation."""
        B = start_grid.shape[0]

        # Encode starting obs
        encoder_out = self.encoder(start_grid, start_scalars)
        h, z = self.rssm.initial_state(B)
        h = h.to(start_grid.device)
        z = z.to(start_grid.device)

        # Get initial posterior
        h_dummy = torch.zeros_like(h)
        z, _ = self.rssm.posterior(h_dummy, encoder_out)

        all_h, all_z, all_reward, all_cont = [], [], [], []

        for t in range(horizon):
            action = action_fn(h, z)
            h = self.rssm.dynamics(h, z, action)
            z, _ = self.rssm.prior(h)

            latent = torch.cat([h, z], dim=-1)
            reward = self.reward_head(latent).squeeze(-1)
            cont = torch.sigmoid(self.continue_head(latent)).squeeze(-1)

            all_h.append(h)
            all_z.append(z)
            all_reward.append(reward)
            all_cont.append(cont)

        return {
            "h": torch.stack(all_h, dim=1),
            "z": torch.stack(all_z, dim=1),
            "reward": torch.stack(all_reward, dim=1),
            "cont": torch.stack(all_cont, dim=1),
        }

    def decode(
        self, h: torch.Tensor, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent state to observation."""
        latent = torch.cat([h, z], dim=-1)
        return self.decoder(latent)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_world_model.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pacman/world_model/world_model.py tests/test_world_model.py
git commit -m "feat(world-model): add integrated WorldModel with train_step and imagine"
```

---

### Task 5: Data Collection Script

**Files:**
- Create: `scripts/collect_data.py`

- [ ] **Step 1: Implement data collection script**

```python
# scripts/collect_data.py
"""Collect gameplay data from a trained PPO agent for world model training."""
import argparse
from pathlib import Path

import numpy as np
import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.training.checkpoint import load_checkpoint
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.replay_buffer import EpisodeReplayBuffer


def collect_episode(
    env: PacmanEnv, network: ActorCritic, device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Play one episode with the trained agent, returning single-frame obs."""
    grids, scalars_list, actions, rewards, dones = [], [], [], [], []

    obs, _ = env.reset(seed=seed)
    while True:
        # Get single-frame observation (not stacked)
        raw_obs = env._build_obs()
        grids.append(raw_obs["grid"])
        scalars_list.append(raw_obs["scalars"])

        # Agent uses stacked obs for action selection
        grid_t = torch.as_tensor(obs["grid"][None], device=device)
        scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
        mask = env.get_legal_mask()
        mask_t = torch.as_tensor(mask[None], device=device)

        with torch.no_grad():
            logits, _ = network(grid_t, scalars_t, mask_t)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)

        if terminated or truncated:
            break

    return {
        "grid": torch.as_tensor(np.stack(grids)),
        "scalars": torch.as_tensor(np.stack(scalars_list)),
        "action": torch.tensor(actions, dtype=torch.long),
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "done": torch.tensor(dones, dtype=torch.bool),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained agent checkpoint (e.g., runs/.../checkpoints/best.pt)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for replay buffer (default: same run dir)")
    parser.add_argument("--difficulty", type=int, default=2)
    args = parser.parse_args()

    config = load_config()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load trained agent
    env_cfg = config["env"]
    net_cfg = config["network"]
    grid_channels = env_cfg["observation_channels"] * env_cfg.get("frame_stack", 1)
    network = ActorCritic(
        grid_channels=grid_channels,
        cnn_channels=net_cfg["cnn_channels"],
        cnn_kernels=net_cfg["cnn_kernels"],
        cnn_strides=net_cfg["cnn_strides"],
        shared_hidden=net_cfg["shared_hidden"],
        head_hidden=net_cfg["head_hidden"],
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    meta = load_checkpoint(ckpt_path.parent, network, filename=ckpt_path.name)
    network.eval()
    print(f"Loaded checkpoint: update={meta['update']}")

    # Collect episodes
    env = PacmanEnv(config, difficulty=args.difficulty)
    buffer = EpisodeReplayBuffer(max_episodes=args.episodes + 100)

    for i in range(args.episodes):
        episode = collect_episode(env, network, device, seed=i)
        buffer.add_episode(episode)
        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1}/{args.episodes} episodes "
                  f"({buffer.total_steps:,} total steps)")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ckpt_path.parent.parent / "replay_buffer.pt"

    buffer.save(out_path)
    print(f"\nSaved {len(buffer)} episodes ({buffer.total_steps:,} steps) to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs (dry run with small episode count)**

Run: `python scripts/collect_data.py --checkpoint runs/2026-04-03_19-27-33/checkpoints/best.pt --episodes 5`
Expected: Prints progress, saves replay_buffer.pt

- [ ] **Step 3: Commit**

```bash
git add scripts/collect_data.py
git commit -m "feat: add data collection script for world model training"
```

---

### Task 6: World Model Training Loop

**Files:**
- Create: `pacman/training/wm_trainer.py`
- Create: `scripts/train_world_model.py`

- [ ] **Step 1: Implement the world model trainer**

```python
# pacman/training/wm_trainer.py
"""World model training loop."""
from pathlib import Path
import time

import torch

from ..world_model.world_model import WorldModel
from ..world_model.replay_buffer import EpisodeReplayBuffer


class WMTrainer:
    """Trains the world model from a replay buffer of collected episodes."""

    def __init__(
        self,
        world_model: WorldModel,
        replay_buffer: EpisodeReplayBuffer,
        device: torch.device,
        lr: float = 3e-4,
        grad_clip: float = 100.0,
        seq_len: int = 50,
        batch_size: int = 16,
    ):
        self.wm = world_model.to(device)
        self.buffer = replay_buffer
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(self.wm.parameters(), lr=lr)

    def train(
        self,
        total_steps: int = 100_000,
        log_every: int = 100,
        save_every: int = 5000,
        save_dir: Path | None = None,
    ) -> None:
        self.wm.train()
        start_time = time.time()

        for step in range(1, total_steps + 1):
            batch = self.buffer.sample_sequences(self.batch_size, self.seq_len)
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            losses = self.wm.train_step(batch)
            total_loss = losses["_total_tensor"]

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip)
            self.optimizer.step()

            if step % log_every == 0:
                elapsed = time.time() - start_time
                sps = step / elapsed
                print(
                    f"[Step {step:>6d}] "
                    f"total={losses['total']:.4f} "
                    f"recon={losses['reconstruction']:.4f} "
                    f"reward={losses['reward']:.4f} "
                    f"cont={losses['continue']:.4f} "
                    f"kl={losses['kl']:.4f} "
                    f"({sps:.1f} steps/s)"
                )

            if save_dir and step % save_every == 0:
                self._save(save_dir, step)

        if save_dir:
            self._save(save_dir, total_steps)
            print(f"Training complete. Final model saved to {save_dir}")

    def _save(self, save_dir: Path, step: int) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.wm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, save_dir / "world_model_latest.pt")
        torch.save({
            "step": step,
            "model_state_dict": self.wm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, save_dir / f"world_model_{step}.pt")
```

- [ ] **Step 2: Implement the training script**

```python
# scripts/train_world_model.py
"""Train world model from collected replay buffer."""
import argparse
from pathlib import Path

import torch

from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.training.wm_trainer import WMTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to replay_buffer.pt")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading replay buffer from {args.data}...")
    buffer = EpisodeReplayBuffer.load(args.data)
    print(f"Loaded {len(buffer)} episodes ({buffer.total_steps:,} steps)")

    # Create world model
    wm = WorldModel()
    total_params = sum(p.numel() for p in wm.parameters())
    print(f"World model parameters: {total_params:,}")

    # Save dir
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(args.data).parent / "world_model"

    # Train
    trainer = WMTrainer(
        world_model=wm,
        replay_buffer=buffer,
        device=device,
        lr=args.lr,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    trainer.train(
        total_steps=args.total_steps,
        log_every=100,
        save_every=5000,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify training starts (100 steps)**

Run: `python scripts/train_world_model.py --data runs/2026-04-03_19-27-33/replay_buffer.pt --total-steps 100 --batch-size 4 --seq-len 10`
Expected: Prints 1 log line at step 100, no errors

- [ ] **Step 4: Commit**

```bash
git add pacman/training/wm_trainer.py scripts/train_world_model.py
git commit -m "feat: add world model training loop and script"
```

---

### Task 7: Dream Agent Training (Imagination PPO)

**Files:**
- Create: `pacman/training/dream_trainer.py`
- Create: `scripts/train_dreamer.py`

- [ ] **Step 1: Implement the dream trainer**

```python
# pacman/training/dream_trainer.py
"""Train a PPO agent entirely inside a learned world model."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..world_model.world_model import WorldModel
from ..env.pacman_env import PacmanEnv
from ..training.evaluator import Evaluator


class DreamPolicy(nn.Module):
    """MLP actor-critic operating on latent state (h, z)."""

    def __init__(self, latent_dim: int = 2560, hidden: int = 512,
                 head_hidden: int = 256, num_actions: int = 4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(latent), self.critic(latent)


class DreamTrainer:
    """Trains a policy purely from world model imagination."""

    def __init__(
        self,
        world_model: WorldModel,
        config: dict,
        device: torch.device,
        imagination_horizon: int = 15,
        num_imaginations: int = 512,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
    ):
        self.wm = world_model.to(device)
        self.wm.eval()  # world model is frozen
        self.config = config
        self.device = device
        self.horizon = imagination_horizon
        self.num_imaginations = num_imaginations
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs

        latent_dim = self.wm.rssm.latent_dim
        self.policy = DreamPolicy(latent_dim=latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Real environment for validation
        self.evaluator = Evaluator(config)

    def _get_starting_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get diverse starting observations from real environment."""
        env = PacmanEnv(self.config, difficulty=2)
        grids, scalars = [], []
        for i in range(self.num_imaginations):
            obs, _ = env.reset(seed=i)
            raw = env._build_obs()
            grids.append(raw["grid"])
            scalars.append(raw["scalars"])
            # Advance a random number of steps for diversity
            import random
            for _ in range(random.randint(0, 200)):
                mask = env.get_legal_mask()
                legal = np.where(mask)[0]
                action = np.random.choice(legal)
                obs, _, done, _, _ = env.step(action)
                if done:
                    obs, _ = env.reset(seed=i + self.num_imaginations)
                    break
            raw = env._build_obs()
            grids[-1] = raw["grid"]
            scalars[-1] = raw["scalars"]

        return (
            torch.as_tensor(np.stack(grids), device=self.device),
            torch.as_tensor(np.stack(scalars), device=self.device),
        )

    def _imagine_rollout(
        self, start_grids: torch.Tensor, start_scalars: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate imagination trajectories."""
        B = start_grids.shape[0]

        with torch.no_grad():
            encoder_out = self.wm.encoder(start_grids, start_scalars)
            h, _ = self.wm.rssm.initial_state(B)
            h = h.to(self.device)
            _, z = self.wm.rssm.posterior(h, encoder_out)

        latents, actions, log_probs, rewards, conts, values = (
            [], [], [], [], [], [],
        )

        for t in range(self.horizon):
            latent = torch.cat([h, z], dim=-1)
            logits, value = self.policy(latent)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            latents.append(latent.detach())
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.squeeze(-1))

            with torch.no_grad():
                h = self.wm.rssm.dynamics(h, z, action)
                z, _ = self.wm.rssm.prior(h)
                lat = torch.cat([h, z], dim=-1)
                reward = self.wm.reward_head(lat).squeeze(-1)
                cont = torch.sigmoid(self.wm.continue_head(lat)).squeeze(-1)

            rewards.append(reward)
            conts.append(cont)

        # Bootstrap value
        with torch.no_grad():
            final_latent = torch.cat([h, z], dim=-1)
            _, bootstrap_value = self.policy(final_latent)
            bootstrap_value = bootstrap_value.squeeze(-1)

        return {
            "latent": torch.stack(latents, dim=1),       # (B, H, D)
            "action": torch.stack(actions, dim=1),        # (B, H)
            "log_prob": torch.stack(log_probs, dim=1),    # (B, H)
            "reward": torch.stack(rewards, dim=1),        # (B, H)
            "cont": torch.stack(conts, dim=1),            # (B, H)
            "value": torch.stack(values, dim=1),          # (B, H)
            "bootstrap_value": bootstrap_value,            # (B,)
        }

    def _compute_gae(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        rewards = rollout["reward"]
        values = rollout["value"]
        conts = rollout["cont"]
        bootstrap = rollout["bootstrap_value"]

        B, H = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=self.device)

        for t in reversed(range(H)):
            next_val = bootstrap if t == H - 1 else values[:, t + 1]
            delta = rewards[:, t] + self.gamma * conts[:, t] * next_val - values[:, t]
            last_gae = delta + self.gamma * self.gae_lambda * conts[:, t] * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, rollout: dict, advantages: torch.Tensor,
                     returns: torch.Tensor) -> dict[str, float]:
        """PPO update on imagination data."""
        B, H, D = rollout["latent"].shape
        flat_latent = rollout["latent"].reshape(B * H, D)
        flat_actions = rollout["action"].reshape(B * H)
        flat_old_log_probs = rollout["log_prob"].reshape(B * H).detach()
        flat_advantages = advantages.reshape(B * H).detach()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        flat_returns = returns.reshape(B * H).detach()

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            logits, values = self.policy(flat_latent)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(flat_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - flat_old_log_probs)
            surr1 = ratio * flat_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * flat_advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            v_loss = F.mse_loss(values.squeeze(-1), flat_returns)

            loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy.item()

        n = self.ppo_epochs
        return {
            "pg_loss": total_pg_loss / n,
            "value_loss": total_v_loss / n,
            "entropy": total_entropy / n,
        }

    def train(
        self,
        total_updates: int = 5000,
        log_every: int = 10,
        eval_every: int = 100,
        eval_episodes: int = 20,
        save_dir: Path | None = None,
    ) -> None:
        start_time = time.time()
        best_score = 0.0

        for update in range(1, total_updates + 1):
            # Get diverse starting states
            start_grids, start_scalars = self._get_starting_states()

            # Imagine rollout
            self.policy.train()
            rollout = self._imagine_rollout(start_grids, start_scalars)

            # Compute advantages
            advantages, returns = self._compute_gae(rollout)

            # PPO update
            losses = self._ppo_update(rollout, advantages, returns)

            if update % log_every == 0:
                elapsed = time.time() - start_time
                mean_reward = rollout["reward"].sum(dim=1).mean().item()
                print(
                    f"[Update {update:>5d}] "
                    f"dream_reward={mean_reward:.1f} "
                    f"pg={losses['pg_loss']:.4f} "
                    f"vl={losses['value_loss']:.4f} "
                    f"ent={losses['entropy']:.3f} "
                    f"({elapsed:.0f}s)"
                )

            if update % eval_every == 0:
                # Evaluate in real environment — this is the key metric
                eval_result = self._evaluate_in_real_env(eval_episodes)
                score = eval_result["mean_score"]
                clear = eval_result["level_clear_rate"]
                print(
                    f"  [REAL EVAL] score={score:.0f} clear={clear:.1%}"
                )
                if score > best_score:
                    best_score = score
                    if save_dir:
                        self._save(save_dir, update, is_best=True)

        if save_dir:
            self._save(save_dir, total_updates, is_best=False)

    def _evaluate_in_real_env(self, num_episodes: int) -> dict:
        """Deploy dream-trained policy in real Pac-Man."""
        self.policy.eval()
        env = PacmanEnv(self.config, difficulty=2)
        scores, cleared = [], 0

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep + 10000)
            # Initialize RSSM state
            h, z = self.wm.rssm.initial_state(1)
            h, z = h.to(self.device), z.to(self.device)

            while True:
                raw = env._build_obs()
                grid_t = torch.as_tensor(raw["grid"][None], device=self.device)
                scalars_t = torch.as_tensor(raw["scalars"][None], device=self.device)

                with torch.no_grad():
                    enc = self.wm.encoder(grid_t, scalars_t)
                    z, _ = self.wm.rssm.posterior(h, enc)
                    latent = torch.cat([h, z], dim=-1)
                    logits, _ = self.policy(latent)
                    # Greedy action
                    action = logits.argmax(dim=-1).item()

                    # Advance RSSM state
                    action_t = torch.tensor([action], device=self.device)
                    h = self.wm.rssm.dynamics(h, z, action_t)

                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    scores.append(info["score"])
                    if info.get("winner") == "pacman":
                        cleared += 1
                    break

        self.policy.train()
        return {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "level_clear_rate": cleared / max(num_episodes, 1),
        }

    def _save(self, save_dir: Path, update: int, is_best: bool = False) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "update": update,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(data, save_dir / "dream_agent_latest.pt")
        if is_best:
            torch.save(data, save_dir / "dream_agent_best.pt")
```

- [ ] **Step 2: Implement the training script**

```python
# scripts/train_dreamer.py
"""Train a PPO agent entirely inside a learned world model."""
import argparse
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.world_model.world_model import WorldModel
from pacman.training.dream_trainer import DreamTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", type=str, required=True,
                        help="Path to world_model_latest.pt")
    parser.add_argument("--total-updates", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-imaginations", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load world model
    wm = WorldModel()
    ckpt = torch.load(args.world_model, map_location="cpu", weights_only=False)
    wm.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded world model from step {ckpt['step']}")

    # Save dir
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(args.world_model).parent / "dream_agent"

    # Train
    trainer = DreamTrainer(
        world_model=wm,
        config=config,
        device=device,
        imagination_horizon=args.horizon,
        num_imaginations=args.num_imaginations,
        lr=args.lr,
    )
    trainer.train(
        total_updates=args.total_updates,
        log_every=10,
        eval_every=100,
        eval_episodes=20,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add pacman/training/dream_trainer.py scripts/train_dreamer.py
git commit -m "feat: add dream agent trainer with imagination PPO and real-env validation"
```

---

### Task 8: Dream Viewer Visualization

**Files:**
- Create: `pacman/viz/dream_viewer.py`
- Create: `scripts/watch_dreams.py`

- [ ] **Step 1: Implement the dream viewer**

```python
# pacman/viz/dream_viewer.py
"""Side-by-side visualization: real game vs. world model dream."""
import numpy as np
import pygame
import torch

from pacman.engine.constants import MAZE_ROWS, MAZE_COLS, Tile, GhostMode, NUM_GHOSTS
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel


# Channel-to-color mapping for the 8-channel grid
CHANNEL_COLORS = [
    (33, 33, 222),    # Ch 0: Walls (blue)
    (255, 255, 0),    # Ch 1: Pac-Man (yellow)
    (255, 184, 151),  # Ch 2: Pellets (peach)
    (255, 184, 255),  # Ch 3: Power pellets (pink)
    (255, 0, 0),      # Ch 4: Dangerous ghosts (red)
    (33, 33, 255),    # Ch 5: Edible ghosts (blue)
    (40, 40, 40),     # Ch 6: Ghost house (dark)
    (0, 255, 0),      # Ch 7: Fruit (green)
]


def grid_tensor_to_surface(
    grid: np.ndarray, tile_size: int,
) -> pygame.Surface:
    """Convert an 8-channel grid (8, 31, 28) to a pygame surface."""
    h, w = MAZE_ROWS * tile_size, MAZE_COLS * tile_size
    surface = pygame.Surface((w, h))
    surface.fill((0, 0, 0))

    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            x, y = c * tile_size, r * tile_size
            # Find the dominant channel
            vals = [grid[ch, r, c] for ch in range(8)]
            max_ch = np.argmax(vals)
            if vals[max_ch] > 0.3:
                color = CHANNEL_COLORS[max_ch]
                alpha = min(1.0, vals[max_ch])
                color = tuple(int(c * alpha) for c in color)
                pygame.draw.rect(surface, color, (x, y, tile_size, tile_size))

    return surface


class DreamViewer:
    """Shows real game and world model reconstruction side-by-side."""

    def __init__(
        self,
        world_model: WorldModel,
        config: dict,
        device: torch.device,
        tile_size: int = 16,
    ):
        self.wm = world_model.to(device)
        self.wm.eval()
        self.config = config
        self.device = device
        self.tile_size = tile_size

        self.maze_w = MAZE_COLS * tile_size
        self.maze_h = MAZE_ROWS * tile_size
        gap = 20
        info_h = 60
        self.width = self.maze_w * 2 + gap
        self.height = self.maze_h + info_h

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dreaming Pac-Man: Real vs Imagined")
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)
        self.clock = pygame.time.Clock()

        self.env = PacmanEnv(config, difficulty=2)
        self.gap = gap

    def run(self, policy_fn=None, max_steps: int = 3000) -> None:
        """Run the viewer.

        Args:
            policy_fn: callable(grid_tensor, scalars_tensor) -> action int.
                       If None, uses random actions.
        """
        obs, _ = self.env.reset(seed=42)
        h, z = self.wm.rssm.initial_state(1)
        h, z = h.to(self.device), z.to(self.device)
        step = 0
        total_divergence = 0.0
        running = True

        while running and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # Get real observation (single frame)
            raw = self.env._build_obs()
            grid_real = raw["grid"]
            scalars_real = raw["scalars"]

            # Encode real observation into world model
            grid_t = torch.as_tensor(grid_real[None], device=self.device)
            scalars_t = torch.as_tensor(scalars_real[None], device=self.device)

            with torch.no_grad():
                enc = self.wm.encoder(grid_t, scalars_t)
                z, _ = self.wm.rssm.posterior(h, enc)
                latent = torch.cat([h, z], dim=-1)

                # Decode world model's reconstruction
                grid_dream, scalars_dream = self.wm.decode(h, z)
                grid_dream_np = grid_dream[0].cpu().numpy()

            # Choose action
            if policy_fn:
                action = policy_fn(grid_t, scalars_t, h, z)
            else:
                mask = self.env.get_legal_mask()
                legal = np.where(mask)[0]
                action = np.random.choice(legal)

            # Advance RSSM
            with torch.no_grad():
                action_t = torch.tensor([action], device=self.device)
                h = self.wm.rssm.dynamics(h, z, action_t)

            # Step real env
            obs, _, terminated, truncated, info = self.env.step(action)
            step += 1

            # Compute divergence
            divergence = np.mean((grid_real - grid_dream_np) ** 2)
            total_divergence += divergence

            # Draw
            self.screen.fill((0, 0, 0))

            # Real game (left)
            real_surface = grid_tensor_to_surface(grid_real, self.tile_size)
            self.screen.blit(real_surface, (0, 0))

            # Dream (right)
            dream_surface = grid_tensor_to_surface(grid_dream_np, self.tile_size)
            self.screen.blit(dream_surface, (self.maze_w + self.gap, 0))

            # Labels
            y = self.maze_h + 5
            self._text("REAL GAME", 10, y, self.font_large, (255, 255, 0))
            self._text(f"Score: {info.get('score', 0)}", 10, y + 20, self.font, (200, 200, 200))
            self._text(f"Step: {step}", 10, y + 36, self.font, (150, 150, 150))

            dx = self.maze_w + self.gap
            self._text("MODEL'S DREAM", dx + 10, y, self.font_large, (100, 200, 255))
            self._text(f"Divergence: {divergence:.4f}", dx + 10, y + 20, self.font, (200, 200, 200))
            avg_div = total_divergence / step
            self._text(f"Avg div: {avg_div:.4f}", dx + 10, y + 36, self.font, (150, 150, 150))

            pygame.display.flip()
            self.clock.tick(10)

            if terminated or truncated:
                obs, _ = self.env.reset(seed=step)
                h, z = self.wm.rssm.initial_state(1)
                h, z = h.to(self.device), z.to(self.device)

        pygame.quit()

    def _text(self, text, x, y, font, color):
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
```

- [ ] **Step 2: Implement the watch script**

```python
# scripts/watch_dreams.py
"""Launch the dream viewer: side-by-side real vs imagined Pac-Man."""
import argparse
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.world_model.world_model import WorldModel
from pacman.viz.dream_viewer import DreamViewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-model", type=str, required=True,
                        help="Path to world_model_latest.pt")
    args = parser.parse_args()

    config = load_config()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load world model
    wm = WorldModel()
    ckpt = torch.load(args.world_model, map_location="cpu", weights_only=False)
    wm.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded world model from step {ckpt['step']}")

    # Launch viewer
    viewer = DreamViewer(world_model=wm, config=config, device=device)
    viewer.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add pacman/viz/dream_viewer.py scripts/watch_dreams.py
git commit -m "feat: add dream viewer for side-by-side real vs imagined visualization"
```

---

### Task 9: End-to-End Integration Test

**Files:**
- Test: `tests/test_dreaming_e2e.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_dreaming_e2e.py
"""End-to-end test: collect data -> train world model -> imagine."""
import torch
import numpy as np
import pytest

from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.world_model.world_model import WorldModel


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def small_buffer(config):
    """Collect 5 short episodes with random actions."""
    env = PacmanEnv(config, difficulty=0)
    buf = EpisodeReplayBuffer(max_episodes=100)

    for i in range(5):
        obs, _ = env.reset(seed=i)
        grids, scalars, actions, rewards, dones = [], [], [], [], []
        for step in range(50):
            raw = env._build_obs()
            grids.append(raw["grid"])
            scalars.append(raw["scalars"])
            mask = env.get_legal_mask()
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break

        buf.add_episode({
            "grid": torch.as_tensor(np.stack(grids)),
            "scalars": torch.as_tensor(np.stack(scalars)),
            "action": torch.tensor(actions, dtype=torch.long),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.bool),
        })
    return buf


class TestDreamingE2E:
    def test_world_model_trains_without_error(self, small_buffer):
        wm = WorldModel()
        optimizer = torch.optim.Adam(wm.parameters(), lr=3e-4)
        wm.train()

        for step in range(3):
            batch = small_buffer.sample_sequences(batch_size=2, seq_len=10)
            losses = wm.train_step(batch)
            total = losses["_total_tensor"]
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        assert losses["total"] > 0
        assert torch.isfinite(torch.tensor(losses["total"]))

    def test_imagination_produces_valid_output(self, small_buffer, config):
        wm = WorldModel()
        wm.eval()

        env = PacmanEnv(config, difficulty=0)
        obs, _ = env.reset(seed=0)
        raw = env._build_obs()
        grid = torch.as_tensor(raw["grid"][None])
        scalars = torch.as_tensor(raw["scalars"][None])

        action_fn = lambda h, z: torch.randint(0, 4, (h.shape[0],))
        imagined = wm.imagine(grid, scalars, action_fn, horizon=5)

        assert imagined["h"].shape == (1, 5, 512)
        assert imagined["reward"].shape == (1, 5)
        assert not torch.isnan(imagined["reward"]).any()

    def test_decode_produces_valid_grid(self, small_buffer):
        wm = WorldModel()
        wm.eval()
        h = torch.randn(1, 512)
        z = torch.randn(1, 2048)
        grid, scalars = wm.decode(h, z)
        assert grid.shape == (1, 8, 31, 28)
        assert not torch.isnan(grid).any()
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_dreaming_e2e.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_dreaming_e2e.py
git commit -m "test: add end-to-end integration tests for dreaming pipeline"
```

---

### Task 10: Run Full Pipeline

This task has no code to write — it runs the full pipeline end-to-end on real data.

- [ ] **Step 1: Collect gameplay data (~30 min)**

Run: `nohup python -u scripts/collect_data.py --checkpoint runs/2026-04-03_19-27-33/checkpoints/best.pt --episodes 5000 > collect.log 2>&1 &`

Monitor: `tail -f collect.log`

Expected: ~15M frames, ~2GB replay_buffer.pt

- [ ] **Step 2: Train world model (~8 hours)**

Run: `nohup python -u scripts/train_world_model.py --data runs/2026-04-03_19-27-33/replay_buffer.pt --total-steps 100000 > wm_training.log 2>&1 &`

Monitor: `tail -f wm_training.log`

Expected: reconstruction loss drops below 0.05, reward accuracy >90%

- [ ] **Step 3: Train dream agent (~2 hours)**

Run: `nohup python -u scripts/train_dreamer.py --world-model runs/2026-04-03_19-27-33/world_model/world_model_latest.pt --total-updates 5000 > dream_training.log 2>&1 &`

Monitor: `tail -f dream_training.log`

Expected: REAL EVAL score increases over time, target >60% of PPO agent score

- [ ] **Step 4: Watch the dreams**

Run: `python scripts/watch_dreams.py --world-model runs/2026-04-03_19-27-33/world_model/world_model_latest.pt`

Expected: Side-by-side window showing real game vs. model reconstruction

- [ ] **Step 5: Commit final results**

```bash
git add -A
git commit -m "feat: complete Dreaming Pac-Man pipeline - world model + imagination training"
```
