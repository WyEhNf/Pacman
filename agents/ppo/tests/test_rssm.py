# tests/test_rssm.py
"""Tests for the RSSM dynamics model."""
import torch
import pytest

from pacman.world_model.rssm import RSSM


BATCH = 4


@pytest.fixture
def rssm():
    return RSSM(
        stoch_classes=32,
        stoch_categoricals=64,
        gru_hidden=512,
        action_dim=4,
        encoder_output_dim=512,
    )


class TestRSSM:
    def test_initial_state_shape(self, rssm):
        h, z = rssm.initial_state(BATCH)
        assert h.shape == (BATCH, 512), f"Expected h shape (4, 512), got {h.shape}"
        assert z.shape == (BATCH, 2048), f"Expected z shape (4, 2048), got {z.shape}"

    def test_dynamics_step(self, rssm):
        h, z = rssm.initial_state(BATCH)
        action = torch.randint(0, 4, (BATCH,))
        h_next = rssm.dynamics(h, z, action)
        assert h_next.shape == (BATCH, 512), f"Expected h_next shape (4, 512), got {h_next.shape}"

    def test_prior(self, rssm):
        h = torch.randn(BATCH, 512)
        z_prior, logits = rssm.prior(h)
        assert z_prior.shape == (BATCH, 2048), f"Expected z_prior shape (4, 2048), got {z_prior.shape}"
        assert logits.shape == (BATCH, 32, 64), f"Expected logits shape (4, 32, 64), got {logits.shape}"

    def test_posterior(self, rssm):
        h = torch.randn(BATCH, 512)
        encoder_out = torch.randn(BATCH, 512)
        z_post, logits = rssm.posterior(h, encoder_out)
        assert z_post.shape == (BATCH, 2048), f"Expected z_post shape (4, 2048), got {z_post.shape}"
        assert logits.shape == (BATCH, 32, 64), f"Expected logits shape (4, 32, 64), got {logits.shape}"

    def test_latent_dim_property(self, rssm):
        assert rssm.latent_dim == 2560, f"Expected latent_dim 2560, got {rssm.latent_dim}"

    def test_full_forward_sequence(self, rssm):
        """Run a 5-step rollout and verify all intermediate shapes."""
        rssm.train()
        h, z = rssm.initial_state(BATCH)
        encoder_out = torch.randn(BATCH, 512)

        for _ in range(5):
            action = torch.randint(0, 4, (BATCH,))
            h = rssm.dynamics(h, z, action)
            z, logits = rssm.posterior(h, encoder_out)

            assert h.shape == (BATCH, 512)
            assert z.shape == (BATCH, 2048)
            assert logits.shape == (BATCH, 32, 64)

    def test_gradients_flow(self, rssm):
        """All parameters should receive gradients through prior + posterior paths."""
        rssm.train()
        h, z = rssm.initial_state(BATCH)
        action = torch.randint(0, 4, (BATCH,))
        encoder_out = torch.randn(BATCH, 512)

        h = rssm.dynamics(h, z, action)
        z_prior, prior_logits = rssm.prior(h)
        z_post, post_logits = rssm.posterior(h, encoder_out)

        loss = z_prior.sum() + prior_logits.sum() + z_post.sum() + post_logits.sum()
        loss.backward()

        for name, p in rssm.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
