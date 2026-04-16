"""Training strategy variants for sim2real transfer experiments.

Variant A: Wider domain randomization + observation delay
Variant B: Pass-through (speed-tuned reward scales set in training script)
Variant C: Advanced 3-mode reset (ground / geometric / reset buffer)
Variant D: A + C combined with speed-tuned reward scales
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional

from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    wrap_to_pi,
    matrix_from_quat,
)

from .quadcopter_strategies import DefaultQuadcopterStrategy

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv


# ---------------------------------------------------------------------------
# Variant A — Wider Domain Randomization + Observation Delay
# ---------------------------------------------------------------------------

class StrategyA(DefaultQuadcopterStrategy):

    def __init__(self, env: QuadcopterEnv):
        super().__init__(env)
        self._obs_delay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_obs_buffer = torch.zeros(self.num_envs, 36, device=self.device)
        self._first_obs_after_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        observations = super().get_observations()
        obs = observations["policy"]

        # Store true current obs before any substitution
        true_obs = obs.clone()

        # For delayed envs (but NOT on first step after reset), use previous obs
        delayed = self._obs_delay_mask & ~self._first_obs_after_reset
        obs[delayed] = self._prev_obs_buffer[delayed]

        self._prev_obs_buffer = true_obs
        self._first_obs_after_reset[:] = False

        observations["policy"] = obs
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        # Run base reset (including default DR)
        super().reset_idx(env_ids)

        n_reset = len(env_ids)

        # --- Overwrite with wider DR ranges ---
        twr_base = self.env._twr_value
        self.env._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(
            twr_base * 0.85, twr_base * 1.15
        )

        k_xy = self.env._k_aero_xy_value
        k_z = self.env._k_aero_z_value
        self.env._K_aero[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
            k_xy * 0.3, k_xy * 3.0
        ).expand(n_reset, 2)
        self.env._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
            k_z * 0.3, k_z * 3.0
        )

        kp_rp = self.env._kp_omega_rp_value
        ki_rp = self.env._ki_omega_rp_value
        kd_rp = self.env._kd_omega_rp_value
        self.env._kp_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
            kp_rp * 0.75, kp_rp * 1.25
        ).expand(n_reset, 2)
        self.env._ki_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
            ki_rp * 0.75, ki_rp * 1.25
        ).expand(n_reset, 2)
        self.env._kd_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
            kd_rp * 0.60, kd_rp * 1.40
        ).expand(n_reset, 2)

        kp_y = self.env._kp_omega_y_value
        ki_y = self.env._ki_omega_y_value
        kd_y = self.env._kd_omega_y_value
        self.env._kp_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
            kp_y * 0.75, kp_y * 1.25
        )
        self.env._ki_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
            ki_y * 0.75, ki_y * 1.25
        )
        self.env._kd_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
            kd_y * 0.60, kd_y * 1.40
        )

        self.env._tau_m[env_ids] = self.env._tau_m_value * torch.empty(
            n_reset, 1, device=self.device
        ).uniform_(0.8, 1.2).expand(n_reset, 4)

        # --- Observation delay mask ---
        self._obs_delay_mask[env_ids] = torch.rand(n_reset, device=self.device) < 0.3
        self._first_obs_after_reset[env_ids] = True


# ---------------------------------------------------------------------------
# Variant B — Speed-Tuned Rewards (pass-through, scales set in training script)
# ---------------------------------------------------------------------------

class StrategyB(DefaultQuadcopterStrategy):
    """Identical to base. Speed incentive comes from reward scale changes only."""
    pass


# ---------------------------------------------------------------------------
# Variant C — Advanced Reset Strategy (ground / geometric / buffer)
# ---------------------------------------------------------------------------

class StrategyC(DefaultQuadcopterStrategy):

    def __init__(self, env: QuadcopterEnv):
        super().__init__(env)
        self._rb_size = 2048
        # Layout: pos_w(3), quat_w(4), lin_vel_w(3), ang_vel_w(3), motor_speeds(4), gate_idx(1) = 18
        self._rb = torch.zeros(self._rb_size, 18, device=self.device)
        self._rb_count = 0
        self._rb_idx = 0

    def get_rewards(self) -> torch.Tensor:
        # Snapshot gate count before base reward (which advances _idx_wp on pass)
        prev_n = self.env._n_gates_passed.clone()

        reward = super().get_rewards()

        # Populate reset buffer with post-gate-crossing states
        newly_passed = self.env._n_gates_passed > prev_n
        ids = torch.where(newly_passed)[0]
        if len(ids) > 0:
            state = torch.cat([
                self.env._robot.data.root_link_pos_w[ids],         # 3
                self.env._robot.data.root_quat_w[ids],             # 4
                self.env._robot.data.root_com_lin_vel_w[ids],      # 3  (world frame)
                self.env._robot.data.root_ang_vel_w[ids],          # 3  (world frame)
                self.env._motor_speeds[ids],                       # 4
                self.env._idx_wp[ids].unsqueeze(1).float(),        # 1  (already advanced)
            ], dim=1)  # (n_new, 18)
            n_new = state.shape[0]
            end = self._rb_idx + n_new
            if end <= self._rb_size:
                self._rb[self._rb_idx:end] = state
            else:
                overflow = end - self._rb_size
                self._rb[self._rb_idx:] = state[:n_new - overflow]
                self._rb[:overflow] = state[n_new - overflow:]
            self._rb_idx = end % self._rb_size
            self._rb_count = min(self._rb_count + n_new, self._rb_size)

        return reward

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # --- Preamble (identical to base) ---
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        self.env._robot.reset(env_ids)

        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]
            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)
            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # --- Three-mode reset (train only) ---
        num_gates = self.env._waypoints.shape[0]
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        if self.cfg.is_train:
            mode = torch.rand(n_reset, device=self.device)
            ground_mask = mode < 0.2
            geo_mask = (mode >= 0.2) & (mode < 0.6)
            buffer_mask = mode >= 0.6

            # Fall back to geometric if buffer is empty
            if self._rb_count == 0:
                geo_mask = geo_mask | buffer_mask
                buffer_mask = buffer_mask & False

            # --- Mode 1: Ground respawn (gate 0) ---
            n_ground = ground_mask.sum().item()
            if n_ground > 0:
                wp_idx = 0
                waypoint_indices[ground_mask] = wp_idx
                x0_wp = self.env._waypoints[wp_idx, 0]
                y0_wp = self.env._waypoints[wp_idx, 1]
                theta = self.env._waypoints[wp_idx, -1]

                x_local = torch.empty(n_ground, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(n_ground, device=self.device).uniform_(-1.0, 1.0)
                cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                x_rot = cos_t * x_local - sin_t * y_local
                y_rot = sin_t * x_local + cos_t * y_local

                default_root_state[ground_mask, 0] = x0_wp - x_rot
                default_root_state[ground_mask, 1] = y0_wp - y_rot
                default_root_state[ground_mask, 2] = 0.1

                yaw = torch.atan2(
                    y0_wp - default_root_state[ground_mask, 1],
                    x0_wp - default_root_state[ground_mask, 0],
                )
                quat = quat_from_euler_xyz(
                    torch.zeros(n_ground, device=self.device),
                    torch.zeros(n_ground, device=self.device),
                    yaw,
                )
                default_root_state[ground_mask, 3:7] = quat
                default_root_state[ground_mask, 7:13] = 0.0  # zero velocity

            # --- Mode 2: Geometric respawn (random gate) ---
            n_geo = geo_mask.sum().item()
            if n_geo > 0:
                wp_rand = torch.randint(0, num_gates, (n_geo,), device=self.device, dtype=self.env._idx_wp.dtype)
                waypoint_indices[geo_mask] = wp_rand

                x0_wp = self.env._waypoints[wp_rand, 0]
                y0_wp = self.env._waypoints[wp_rand, 1]
                z_wp = self.env._waypoints[wp_rand, 2]
                theta = self.env._waypoints[wp_rand, -1]

                x_local = torch.empty(n_geo, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(n_geo, device=self.device).uniform_(-1.0, 1.0)
                z_local = torch.empty(n_geo, device=self.device).uniform_(-0.5, 0.5)

                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                x_rot = cos_t * x_local - sin_t * y_local
                y_rot = sin_t * x_local + cos_t * y_local

                default_root_state[geo_mask, 0] = x0_wp - x_rot
                default_root_state[geo_mask, 1] = y0_wp - y_rot
                default_root_state[geo_mask, 2] = (z_wp + z_local).clamp(min=0.3)

                yaw = torch.atan2(
                    y0_wp - default_root_state[geo_mask, 1],
                    x0_wp - default_root_state[geo_mask, 0],
                )
                yaw_noise = torch.empty(n_geo, device=self.device).uniform_(-0.3, 0.3)
                quat = quat_from_euler_xyz(
                    torch.zeros(n_geo, device=self.device),
                    torch.zeros(n_geo, device=self.device),
                    yaw + yaw_noise,
                )
                default_root_state[geo_mask, 3:7] = quat
                default_root_state[geo_mask, 7:10] = torch.empty(n_geo, 3, device=self.device).uniform_(-0.5, 0.5)
                default_root_state[geo_mask, 10:13] = 0.0

            # --- Mode 3: Reset buffer ---
            n_buf = buffer_mask.sum().item()
            if n_buf > 0:
                sample_idx = torch.randint(0, self._rb_count, (n_buf,), device=self.device)
                sampled = self._rb[sample_idx]  # (n_buf, 18)

                # Perturbed position
                pos = sampled[:, 0:3] + torch.empty(n_buf, 3, device=self.device).uniform_(-0.2, 0.2)
                pos[:, 2] = pos[:, 2].clamp(min=0.2)
                default_root_state[buffer_mask, 0:3] = pos

                # Perturbed orientation: add noise to euler angles
                quat_buf = sampled[:, 3:7]
                rpy = euler_xyz_from_quat(quat_buf)
                roll_noisy = rpy[0] + torch.empty(n_buf, device=self.device).uniform_(-0.1, 0.1)
                pitch_noisy = rpy[1] + torch.empty(n_buf, device=self.device).uniform_(-0.1, 0.1)
                yaw_noisy = rpy[2] + torch.empty(n_buf, device=self.device).uniform_(-0.1, 0.1)
                quat_noisy = quat_from_euler_xyz(roll_noisy, pitch_noisy, yaw_noisy)
                default_root_state[buffer_mask, 3:7] = quat_noisy

                # Perturbed velocity (world frame)
                default_root_state[buffer_mask, 7:10] = sampled[:, 7:10] + torch.empty(
                    n_buf, 3, device=self.device
                ).uniform_(-0.3, 0.3)
                default_root_state[buffer_mask, 10:13] = sampled[:, 10:13]

                # Restore motor speeds
                buf_env_ids = env_ids[buffer_mask]
                self.env._motor_speeds[buf_env_ids] = sampled[:, 13:17]

                # Stored _idx_wp (already advanced to next target)
                waypoint_indices[buffer_mask] = sampled[:, 17].long()
        else:
            # Play mode — same as base
            waypoint_indices[:] = self.env._initial_wp

        # --- Domain randomization (same as base) ---
        if self.cfg.is_train:
            twr_base = self.env._twr_value
            self.env._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(
                twr_base * 0.95, twr_base * 1.05
            )

            k_xy = self.env._k_aero_xy_value
            k_z = self.env._k_aero_z_value
            self.env._K_aero[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                k_xy * 0.5, k_xy * 2.0
            ).expand(n_reset, 2)
            self.env._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                k_z * 0.5, k_z * 2.0
            )

            kp_rp = self.env._kp_omega_rp_value
            ki_rp = self.env._ki_omega_rp_value
            kd_rp = self.env._kd_omega_rp_value
            self.env._kp_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                kp_rp * 0.85, kp_rp * 1.15
            ).expand(n_reset, 2)
            self.env._ki_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                ki_rp * 0.85, ki_rp * 1.15
            ).expand(n_reset, 2)
            self.env._kd_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                kd_rp * 0.7, kd_rp * 1.3
            ).expand(n_reset, 2)

            kp_y = self.env._kp_omega_y_value
            ki_y = self.env._ki_omega_y_value
            kd_y = self.env._kd_omega_y_value
            self.env._kp_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                kp_y * 0.85, kp_y * 1.15
            )
            self.env._ki_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                ki_y * 0.85, ki_y * 1.15
            )
            self.env._kd_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                kd_y * 0.7, kd_y * 1.3
            )

            self.env._tau_m[env_ids] = self.env._tau_m_value * torch.empty(
                n_reset, 1, device=self.device
            ).uniform_(0.9, 1.1).expand(n_reset, 4)

        # --- Play mode position override ---
        if not self.cfg.is_train:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices[:] = self.env._initial_wp

        # --- Shared tail (identical to base) ---
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        self.env._crashed[env_ids] = 0


# ---------------------------------------------------------------------------
# Variant D — All Combined (A + C, with speed reward scales from training script)
# ---------------------------------------------------------------------------

class StrategyD(DefaultQuadcopterStrategy):

    def __init__(self, env: QuadcopterEnv):
        super().__init__(env)

        # From A: observation delay
        self._obs_delay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_obs_buffer = torch.zeros(self.num_envs, 36, device=self.device)
        self._first_obs_after_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # From C: reset buffer
        self._rb_size = 2048
        self._rb = torch.zeros(self._rb_size, 18, device=self.device)
        self._rb_count = 0
        self._rb_idx = 0

    # --- From A: observation delay ---
    def get_observations(self) -> Dict[str, torch.Tensor]:
        observations = super().get_observations()
        obs = observations["policy"]

        true_obs = obs.clone()
        delayed = self._obs_delay_mask & ~self._first_obs_after_reset
        obs[delayed] = self._prev_obs_buffer[delayed]

        self._prev_obs_buffer = true_obs
        self._first_obs_after_reset[:] = False

        observations["policy"] = obs
        return observations

    # --- From C: buffer population ---
    def get_rewards(self) -> torch.Tensor:
        prev_n = self.env._n_gates_passed.clone()
        reward = super().get_rewards()

        newly_passed = self.env._n_gates_passed > prev_n
        ids = torch.where(newly_passed)[0]
        if len(ids) > 0:
            state = torch.cat([
                self.env._robot.data.root_link_pos_w[ids],
                self.env._robot.data.root_quat_w[ids],
                self.env._robot.data.root_com_lin_vel_w[ids],
                self.env._robot.data.root_ang_vel_w[ids],
                self.env._motor_speeds[ids],
                self.env._idx_wp[ids].unsqueeze(1).float(),
            ], dim=1)
            n_new = state.shape[0]
            end = self._rb_idx + n_new
            if end <= self._rb_size:
                self._rb[self._rb_idx:end] = state
            else:
                overflow = end - self._rb_size
                self._rb[self._rb_idx:] = state[:n_new - overflow]
                self._rb[:overflow] = state[n_new - overflow:]
            self._rb_idx = end % self._rb_size
            self._rb_count = min(self._rb_count + n_new, self._rb_size)

        return reward

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Combines C's 3-mode reset with A's wider DR + obs delay."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # --- Preamble (identical to base) ---
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        self.env._robot.reset(env_ids)

        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]
            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)
            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # --- From C: Three-mode reset ---
        num_gates = self.env._waypoints.shape[0]
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        if self.cfg.is_train:
            mode = torch.rand(n_reset, device=self.device)
            ground_mask = mode < 0.2
            geo_mask = (mode >= 0.2) & (mode < 0.6)
            buffer_mask = mode >= 0.6

            if self._rb_count == 0:
                geo_mask = geo_mask | buffer_mask
                buffer_mask = buffer_mask & False

            # Mode 1: Ground respawn
            n_ground = ground_mask.sum().item()
            if n_ground > 0:
                wp_idx = 0
                waypoint_indices[ground_mask] = wp_idx
                x0_wp = self.env._waypoints[wp_idx, 0]
                y0_wp = self.env._waypoints[wp_idx, 1]
                theta = self.env._waypoints[wp_idx, -1]

                x_local = torch.empty(n_ground, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(n_ground, device=self.device).uniform_(-1.0, 1.0)
                cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                x_rot = cos_t * x_local - sin_t * y_local
                y_rot = sin_t * x_local + cos_t * y_local

                default_root_state[ground_mask, 0] = x0_wp - x_rot
                default_root_state[ground_mask, 1] = y0_wp - y_rot
                default_root_state[ground_mask, 2] = 0.1

                yaw = torch.atan2(
                    y0_wp - default_root_state[ground_mask, 1],
                    x0_wp - default_root_state[ground_mask, 0],
                )
                quat = quat_from_euler_xyz(
                    torch.zeros(n_ground, device=self.device),
                    torch.zeros(n_ground, device=self.device),
                    yaw,
                )
                default_root_state[ground_mask, 3:7] = quat
                default_root_state[ground_mask, 7:13] = 0.0

            # Mode 2: Geometric respawn
            n_geo = geo_mask.sum().item()
            if n_geo > 0:
                wp_rand = torch.randint(0, num_gates, (n_geo,), device=self.device, dtype=self.env._idx_wp.dtype)
                waypoint_indices[geo_mask] = wp_rand

                x0_wp = self.env._waypoints[wp_rand, 0]
                y0_wp = self.env._waypoints[wp_rand, 1]
                z_wp = self.env._waypoints[wp_rand, 2]
                theta = self.env._waypoints[wp_rand, -1]

                x_local = torch.empty(n_geo, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(n_geo, device=self.device).uniform_(-1.0, 1.0)
                z_local = torch.empty(n_geo, device=self.device).uniform_(-0.5, 0.5)

                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                x_rot = cos_t * x_local - sin_t * y_local
                y_rot = sin_t * x_local + cos_t * y_local

                default_root_state[geo_mask, 0] = x0_wp - x_rot
                default_root_state[geo_mask, 1] = y0_wp - y_rot
                default_root_state[geo_mask, 2] = (z_wp + z_local).clamp(min=0.3)

                yaw = torch.atan2(
                    y0_wp - default_root_state[geo_mask, 1],
                    x0_wp - default_root_state[geo_mask, 0],
                )
                yaw_noise = torch.empty(n_geo, device=self.device).uniform_(-0.3, 0.3)
                quat = quat_from_euler_xyz(
                    torch.zeros(n_geo, device=self.device),
                    torch.zeros(n_geo, device=self.device),
                    yaw + yaw_noise,
                )
                default_root_state[geo_mask, 3:7] = quat
                default_root_state[geo_mask, 7:10] = torch.empty(n_geo, 3, device=self.device).uniform_(-0.5, 0.5)
                default_root_state[geo_mask, 10:13] = 0.0

            # Mode 3: Reset buffer
            n_buf = buffer_mask.sum().item()
            if n_buf > 0:
                sample_idx = torch.randint(0, self._rb_count, (n_buf,), device=self.device)
                sampled = self._rb[sample_idx]

                pos = sampled[:, 0:3] + torch.empty(n_buf, 3, device=self.device).uniform_(-0.2, 0.2)
                pos[:, 2] = pos[:, 2].clamp(min=0.2)
                default_root_state[buffer_mask, 0:3] = pos

                quat_buf = sampled[:, 3:7]
                rpy = euler_xyz_from_quat(quat_buf)
                roll_noisy = rpy[0] + torch.empty(n_buf, device=self.device).uniform_(-0.1, 0.1)
                pitch_noisy = rpy[1] + torch.empty(n_buf, device=self.device).uniform_(-0.1, 0.1)
                yaw_noisy = rpy[2] + torch.empty(n_buf, device=self.device).uniform_(-0.1, 0.1)
                quat_noisy = quat_from_euler_xyz(roll_noisy, pitch_noisy, yaw_noisy)
                default_root_state[buffer_mask, 3:7] = quat_noisy

                default_root_state[buffer_mask, 7:10] = sampled[:, 7:10] + torch.empty(
                    n_buf, 3, device=self.device
                ).uniform_(-0.3, 0.3)
                default_root_state[buffer_mask, 10:13] = sampled[:, 10:13]

                buf_env_ids = env_ids[buffer_mask]
                self.env._motor_speeds[buf_env_ids] = sampled[:, 13:17]

                waypoint_indices[buffer_mask] = sampled[:, 17].long()
        else:
            waypoint_indices[:] = self.env._initial_wp

        # --- From A: Wider DR ---
        if self.cfg.is_train:
            twr_base = self.env._twr_value
            self.env._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(
                twr_base * 0.85, twr_base * 1.15
            )

            k_xy = self.env._k_aero_xy_value
            k_z = self.env._k_aero_z_value
            self.env._K_aero[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                k_xy * 0.3, k_xy * 3.0
            ).expand(n_reset, 2)
            self.env._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                k_z * 0.3, k_z * 3.0
            )

            kp_rp = self.env._kp_omega_rp_value
            ki_rp = self.env._ki_omega_rp_value
            kd_rp = self.env._kd_omega_rp_value
            self.env._kp_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                kp_rp * 0.75, kp_rp * 1.25
            ).expand(n_reset, 2)
            self.env._ki_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                ki_rp * 0.75, ki_rp * 1.25
            ).expand(n_reset, 2)
            self.env._kd_omega[env_ids, :2] = torch.empty(n_reset, 1, device=self.device).uniform_(
                kd_rp * 0.60, kd_rp * 1.40
            ).expand(n_reset, 2)

            kp_y = self.env._kp_omega_y_value
            ki_y = self.env._ki_omega_y_value
            kd_y = self.env._kd_omega_y_value
            self.env._kp_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                kp_y * 0.75, kp_y * 1.25
            )
            self.env._ki_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                ki_y * 0.75, ki_y * 1.25
            )
            self.env._kd_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
                kd_y * 0.60, kd_y * 1.40
            )

            self.env._tau_m[env_ids] = self.env._tau_m_value * torch.empty(
                n_reset, 1, device=self.device
            ).uniform_(0.8, 1.2).expand(n_reset, 4)

        # --- Play mode position override ---
        if not self.cfg.is_train:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices[:] = self.env._initial_wp

        # --- Shared tail ---
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        self.env._crashed[env_ids] = 0

        # --- From A: observation delay state ---
        self._obs_delay_mask[env_ids] = torch.rand(n_reset, device=self.device) < 0.3
        self._first_obs_after_reset[env_ids] = True
