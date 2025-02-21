"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def predict(self, u):

        G = self.calculate_jacobian(self._state.mu.flatten(), u)
        V = self.calculate_V(self._state.mu.flatten(), u)
        self._state_bar.mu = get_prediction(self._state.mu.flatten(), u)[:, np.newaxis]

        # G = self.calculate_jacobian(self._state_bar.mu.flatten(), u)
        # V = self.calculate_V(self._state_bar.mu.flatten(), u)

        self._state_bar.Sigma = G @ self._state.Sigma @ G.T + V @ get_motion_noise_covariance(u, self._alphas) @ V.T



    def calculate_jacobian(self, x, u):
        theta = x[2]
        drot1, dtran, _ = u
        G = np.array([
            [1, 0, -dtran * np.sin(theta + drot1)],
            [0, 1, dtran * np.cos(theta + drot1)],
            [0, 0, 1]
        ])
        return G
    
    def calculate_V(self, x, u):
        theta = x[2]
        drot1, dtran, _ = u
        V = np.array([
            [-dtran * np.sin(theta + drot1), np.cos(theta + drot1),0],
            [dtran * np.cos(theta + drot1), np.sin(theta + drot1),0],
            [1, 0, 1]
        ])
        return V

    def calculate_H(self, x, z):
        m_x, m_y = self._field_map.landmarks_poses_x[int(z[1])], self._field_map.landmarks_poses_y[int(z[1])]
        H = np.array([
            [(m_y - x[1]) / ((m_x - x[0])**2 + (m_y - x[1])**2),
            -(m_x - x[0]) / ((m_x - x[0])**2 + (m_y - x[1])**2),
            -1]
        ])
        return H

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
        H = self.calculate_H(self._state_bar.mu.flatten(), z)
        Q = np.array([[self._Q]])
        S = H @ self._state_bar.Sigma @ H.T + Q
        K = self._state_bar.Sigma @ H.T @ np.linalg.inv(S)

        innovation = wrap_angle(z[0] - get_expected_observation(self._state_bar.mu.flatten(), int(z[1]))[0])
        self._state.mu = self._state_bar.mu + K @ np.array([[innovation]])
        self._state.Sigma = (np.eye(3) - K @ H) @ self._state_bar.Sigma