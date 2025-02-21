"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, bearing_std, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, bearing_std)
        print(f"Initializing PF with {num_particles} particles")
        self._num_particles = num_particles
        self._global_localization = global_localization
        self._particles = np.random.multivariate_normal(initial_state.mu.flatten(), initial_state.Sigma, num_particles)
        self._weights = np.ones(num_particles) / num_particles

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for i in range(self._num_particles):
            self._particles[i] = sample_from_odometry(self._particles[i], u, self._alphas)

        stats = get_gaussian_statistics(self._particles)
        self._state_bar.mu = stats.mu
        self._state_bar.Sigma = stats.Sigma


    def low_variance_resample(self):
        M = self._num_particles
        new_particles = np.zeros_like(self._particles)
        
        r = uniform(0, 1 / M)
        c = self._weights[0]
        i = 0

        for m in range(M):
            u = r + m / M
            while u > c and i < M - 1:
                i += 1
                c += self._weights[i]
            new_particles[m] = self._particles[i]

        self._particles = new_particles
        self._weights.fill(1.0 / M)

    def update(self, z):

        for i in range(self._num_particles):
            expected_observation = get_observation(self._particles[i], int(z[1]))
            innovation = wrap_angle(z[0] - expected_observation[0])
            self._weights[i] = self._weights[i] * gaussian.pdf(innovation, 0, np.sqrt(self._Q))
        
        self._weights = self._weights / np.sum(self._weights)

        self.low_variance_resample()

        stats = get_gaussian_statistics(self._particles)
        self._state.mu = stats.mu
        self._state.Sigma = stats.Sigma
