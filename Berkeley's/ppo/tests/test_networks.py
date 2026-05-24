# tests/test_networks.py
import torch
import pytest

from pacman.agents.networks import ActorCritic


@pytest.fixture
def network():
    return ActorCritic()


class TestActorCritic:
    def test_forward_shapes(self, network):
        grid = torch.randn(2, 8, 31, 28)
        scalars = torch.randn(2, 5)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits, value = network(grid, scalars, mask)
        assert logits.shape == (2, 4)
        assert value.shape == (2, 1)

    def test_softmax_sums_to_one(self, network):
        grid = torch.randn(3, 8, 31, 28)
        scalars = torch.randn(3, 5)
        mask = torch.ones(3, 4, dtype=torch.bool)
        logits, _ = network(grid, scalars, mask)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_action_masking(self, network):
        grid = torch.randn(1, 8, 31, 28)
        scalars = torch.randn(1, 5)
        mask = torch.tensor([[True, False, False, True]])
        logits, _ = network(grid, scalars, mask)
        probs = torch.softmax(logits, dim=-1)
        # Masked actions should have near-zero probability
        assert probs[0, 1].item() < 1e-6
        assert probs[0, 2].item() < 1e-6

    def test_parameter_count(self, network):
        total = sum(p.numel() for p in network.parameters())
        # Should be approximately 1.65M
        assert 1_000_000 < total < 3_000_000

    def test_gradients_flow(self, network):
        grid = torch.randn(2, 8, 31, 28)
        scalars = torch.randn(2, 5)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits, value = network(grid, scalars, mask)
        loss = logits.sum() + value.sum()
        loss.backward()
        for p in network.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
