"""Particle Filter for ghost position tracking (lightweight, real-time)."""
import numpy as np

class GhostParticleFilter:
    def __init__(self, n_particles=100):
        self.n_particles = n_particles
        self.particles = []
    def initialize_uniform(self, legal_positions): raise NotImplementedError
    def predict(self, transition_model): raise NotImplementedError
    def update(self, noisy_distance, pacman_pos, jail_pos, obs_model): raise NotImplementedError
    def get_belief_heatmap(self, height, width) -> np.ndarray: raise NotImplementedError
