# tests/test_world_model.py
"""Tests for the integrated WorldModel."""
import torch
import pytest

from pacman.world_model.world_model import WorldModel


@pytest.fixture
def wm():
    return WorldModel()


class TestWorldModel:
    def test_train_step_returns_losses(self, wm):
        """Pass batch with B=4, T=20. Check all loss keys exist and are finite."""
        B, T = 4, 20
        batch = {
            "grid": torch.randn(B, T, 8, 31, 28),
            "scalars": torch.randn(B, T, 5),
            "action": torch.randint(0, 4, (B, T)),
            "reward": torch.randn(B, T),
            "done": torch.zeros(B, T, dtype=torch.bool),
        }
        wm.train()
        losses = wm.train_step(batch)

        expected_keys = {"recon", "reward", "continue", "kl", "total"}
        assert expected_keys.issubset(losses.keys()), (
            f"Missing loss keys: {expected_keys - losses.keys()}"
        )

        for key in expected_keys:
            assert torch.isfinite(torch.tensor(losses[key])), (
                f"Loss '{key}' is not finite: {losses[key]}"
            )

        # _total_tensor should be a differentiable tensor
        assert "_total_tensor" in losses
        assert isinstance(losses["_total_tensor"], torch.Tensor)
        assert losses["_total_tensor"].requires_grad

    def test_imagine_shapes(self, wm):
        """Start from B=8 observations. Random action_fn. horizon=10."""
        B, horizon = 8, 10
        wm.eval()

        start_grid = torch.randn(B, 8, 31, 28)
        start_scalars = torch.randn(B, 5)

        def random_action_fn(h, z):
            return torch.randint(0, 4, (h.shape[0],))

        result = wm.imagine(start_grid, start_scalars, random_action_fn, horizon=horizon)

        assert result["h"].shape == (B, horizon, 512), (
            f"Expected h shape ({B},{horizon},512), got {result['h'].shape}"
        )
        assert result["z"].shape == (B, horizon, 2048), (
            f"Expected z shape ({B},{horizon},2048), got {result['z'].shape}"
        )
        assert result["reward"].shape == (B, horizon), (
            f"Expected reward shape ({B},{horizon}), got {result['reward'].shape}"
        )
        assert result["cont"].shape == (B, horizon), (
            f"Expected cont shape ({B},{horizon}), got {result['cont'].shape}"
        )

    def test_decode(self, wm):
        """Random h (4,512), z (4,2048). Check grid (4,8,31,28), scalars (4,5)."""
        h = torch.randn(4, 512)
        z = torch.randn(4, 2048)
        grid, scalars = wm.decode(h, z)
        assert grid.shape == (4, 8, 31, 28), f"Expected grid (4,8,31,28), got {grid.shape}"
        assert scalars.shape == (4, 5), f"Expected scalars (4,5), got {scalars.shape}"

    def test_parameter_count(self, wm):
        """Total params between 5M and 50M (encoder ~6M, RSSM ~5M, decoder ~13M, heads ~3M)."""
        total_params = sum(p.numel() for p in wm.parameters())
        assert 5_000_000 < total_params < 50_000_000, (
            f"Parameter count {total_params:,} not in range (5M, 50M)"
        )
