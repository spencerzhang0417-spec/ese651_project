# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """Compute per-timestep rewards that encourage fast gate-to-gate racing."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure

        # --- Gate traversal detection via sign change in gate-frame x ---
        x_gate = self.env._pose_drone_wrt_gate[:, 0]
        yz_dist = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:], dim=1)

        gate_crossed = (self.env._prev_x_drone_wrt_gate > 0) & (x_gate <= 0)
        close_to_center = yz_dist < 0.75
        gate_passed = (gate_crossed & close_to_center).float()

        # backward gate passing penalty
        backward_crossed = (self.env._prev_x_drone_wrt_gate < 0) & (x_gate >= 0) & (yz_dist < 0.75)
        backward_penalty = backward_crossed.float()  # scale this in reward dict

        self.env._prev_x_drone_wrt_gate = x_gate.clone()

        ids_gate_passed = torch.where(gate_passed > 0.5)[0]
        self.env._n_gates_passed[ids_gate_passed] += 1
        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        # Recompute gate-frame pose for newly advanced gates so progress uses the new target
        if len(ids_gate_passed) > 0:
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed],
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = self.env._pose_drone_wrt_gate[ids_gate_passed, 0]

        # --- Progress toward current gate ---
        distance_to_goal = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        delta_distance = self.env._last_distance_to_goal - distance_to_goal
        self.env._last_distance_to_goal = distance_to_goal.clone()
        progress = torch.clamp(delta_distance, -1.0, 1.0)

        # --- Speed toward current gate (split into approach and exit phases) ---
        direction_to_gate = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        dist_to_gate = torch.linalg.norm(direction_to_gate, dim=1, keepdim=True)
        direction_to_gate = direction_to_gate / (dist_to_gate + 1e-8)
        vel_world = self.env._robot.data.root_com_lin_vel_w
        speed_toward_gate = torch.sum(vel_world * direction_to_gate, dim=1)
        dist_scalar = dist_to_gate.squeeze(1)

        # Approach: reward high speed when far from gate (>2m)
        approach_blend = torch.clamp((dist_scalar - 1.0) / 1.0, 0.0, 1.0)
        approach_speed = approach_blend * torch.clamp(speed_toward_gate, 0.0, 8.0) / 8.0

        # Exit: reward moderate, controlled speed when close to gate (<2m)
        exit_blend = 1.0 - approach_blend
        exit_speed = exit_blend * torch.clamp(speed_toward_gate, 0.0, 3.0) / 3.0

        # --- Gate proximity: reward being close to the gate center ---
        gate_proximity = torch.clamp(1.0 - dist_scalar / 3.0, 0.0, 1.0)

        # Only active when targeting gate 2 or gate 3
        is_powerloop = ((self.env._idx_wp == 2) | (self.env._idx_wp == 3)).float()

        # --- Crash detection ---
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "gate_passed": gate_passed * self.env.rew['gate_passed_reward_scale'],
                "approach_speed": approach_speed * self.env.rew['approach_speed_reward_scale'],
                "exit_speed": exit_speed * self.env.rew['exit_speed_reward_scale'],
                "gate_proximity": gate_proximity * self.env.rew['gate_proximity_reward_scale'],
                "time_penalty": torch.ones(self.num_envs, device=self.device) * self.env.rew['time_penalty_reward_scale'],
                "backward_cross": backward_penalty * self.env.rew['backward_cross_reward_scale'],
                "correct_approach": correct_approach * self.env.rew['correct_approach_reward_scale'],
                # "altitude_bonus": altitude_bonus * self.env.rew['altitude_bonus_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Build observation vector for the racing policy.

        Frame choices:
          - Body-frame velocities and angular rates (ego-centric, invariant to world yaw)
          - Body-frame relative gate positions (the policy needs to know where gates are
            relative to its own heading, not in absolute world coordinates)
          - Gate-frame drone position (useful for fine-grained gate alignment)
        """

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations

        # --- Drone state in body frame ---
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b       # (num_envs, 3)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b           # (num_envs, 3)
        drone_quat_w = self.env._robot.data.root_quat_w                 # (num_envs, 4)

        # --- Current gate: relative position in body frame ---
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]
        gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            drone_quat_w,
            current_gate_pos_w,
        )

        # --- Current gate: drone position in gate frame (for alignment) ---
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate            # (num_envs, 3)

        # --- Next gate: relative position in body frame ---
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            drone_quat_w,
            next_gate_pos_w,
        )

        # --- Previous actions (proprioceptive feedback) ---
        prev_actions = self.env._previous_actions                       # (num_envs, 4)

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_lin_vel_b,        # body-frame linear velocity       (3)
                drone_ang_vel_b,        # body-frame angular velocity      (3)
                drone_quat_w,           # orientation quaternion            (4)
                gate_pos_b,             # current gate in body frame       (3)
                drone_pos_gate_frame,   # drone position in gate frame     (3)
                next_gate_pos_b,        # next gate in body frame          (3)
                prev_actions,           # previous actions                 (4)
            ],                          #                          total = 23
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
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

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.

        num_gates = self.env._waypoints.shape[0]

        # Randomly spawn behind any gate so the policy learns the full course
        waypoint_indices = torch.randint(0, num_gates, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)

        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]

        # Randomized offset behind the gate: 1–3m back, ±0.5m lateral, ±0.3m vertical
        x_local = -torch.empty(n_reset, device=self.device).uniform_(1.0, 3.0)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = (z_wp + z_local).clamp(min=0.3)

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Point toward gate with randomized yaw perturbation
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise,
        )
        default_root_state[:, 3:7] = quat

        # Small random initial velocity for robustness
        default_root_state[:, 7:10] = torch.empty(n_reset, 3, device=self.device).uniform_(-0.5, 0.5)

        # --- Domain randomization to bridge the sim2real gap ---
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

        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0