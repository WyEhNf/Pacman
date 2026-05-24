# tests/test_world_model_components.py
"""Tests for world model encoder, decoder, and heads."""
import torch
import pytest

from pacman.world_model.encoder import ObservationEncoder
from pacman.world_model.decoder import ObservationDecoder
from pacman.world_model.heads import RewardHead, ContinueHead


BATCH = 4
LATENT_DIM = 2560


class TestObservationEncoder:
    def test_output_shape(self):
        encoder = ObservationEncoder()
        grid = torch.randn(BATCH, 8, 31, 28)
        scalars = torch.randn(BATCH, 5)
        out = encoder(grid, scalars)
        assert out.shape == (BATCH, 512), f"Expected (4, 512), got {out.shape}"

    def test_gradients_flow(self):
        encoder = ObservationEncoder()
        grid = torch.randn(BATCH, 8, 31, 28)
        scalars = torch.randn(BATCH, 5)
        out = encoder(grid, scalars)
        loss = out.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"


class TestObservationDecoder:
    def test_output_shape(self):
        decoder = ObservationDecoder()
        latent = torch.randn(BATCH, LATENT_DIM)
        grid, scalars = decoder(latent)
        assert grid.shape == (BATCH, 8, 31, 28), f"Expected (4,8,31,28), got {grid.shape}"
        assert scalars.shape == (BATCH, 5), f"Expected (4,5), got {scalars.shape}"

    def test_gradients_flow(self):
        decoder = ObservationDecoder()
        latent = torch.randn(BATCH, LATENT_DIM)
        grid, scalars = decoder(latent)
        loss = grid.sum() + scalars.sum()
        loss.backward()
        for name, p in decoder.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"


class TestHeads:
    def test_reward_head_shape(self):
        head = RewardHead(latent_dim=LATENT_DIM)
        latent = torch.randn(BATCH, LATENT_DIM)
        out = head(latent)
        assert out.shape == (BATCH, 1), f"Expected (4, 1), got {out.shape}"

    def test_continue_head_shape(self):
        head = ContinueHead(latent_dim=LATENT_DIM)
        latent = torch.randn(BATCH, LATENT_DIM)
        out = head(latent)
        assert out.shape == (BATCH, 1), f"Expected (4, 1), got {out.shape}"

    def test_continue_head_outputs_logits(self):
        """ContinueHead should output raw logits (not bounded to [0,1])."""
        head = ContinueHead(latent_dim=LATENT_DIM)
        # Use a large latent to encourage large outputs
        latent = torch.randn(BATCH, LATENT_DIM) * 10.0
        out = head(latent)
        # Logits should be able to go outside [0, 1]
        has_outside = (out > 1.0).any() or (out < 0.0).any()
        assert has_outside, (
            "ContinueHead appears to be applying sigmoid — expected raw logits outside [0, 1]"
        )
