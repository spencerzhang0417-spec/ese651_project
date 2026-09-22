# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL checkpoint: run N trials and report success rate + time stats.

Success = env terminated via time_out (lap-completion) without being `reset_terminated` (no crash).
Failure = any crash / backward gate / altitude violation / ran out of 45s wall time.

The key subtlety: when an env is done, `base._n_gates_passed` is auto-reset to 0 before we can
read it. So we snapshot it BEFORE each step and use `reset_terminated` as the authoritative
crash signal.
"""

"""Launch Isaac Sim Simulator first."""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint over N trials.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Race-v0", help="Task name.")
parser.add_argument("--num_trials", type=int, default=100, help="Number of trials.")
parser.add_argument("--seed", type=int, default=0, help="Base seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--no_plot", action="store_true", default=False, help="Skip matplotlib plot.")
parser.add_argument("--plot_out", type=str, default=None, help="Path to save plot PNG (defaults to log dir).")

# --load_run / --checkpoint / --resume are added by cli_args.add_rsl_rl_args below
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.load_run is None:
    parser.error("--load_run is required (e.g. --load_run 2026-04-18_19-28-47)")
if args_cli.checkpoint is None:
    args_cli.checkpoint = "best_model.pt"

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import src.isaac_quad_sim2real.tasks   # noqa: F401


def main():
    args_cli.resume = True

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_trials,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.is_train = False
    env_cfg.seed = args_cli.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    device = base.device
    num_trials = args_cli.num_trials
    dt_step = base.cfg.sim.dt * base.cfg.decimation
    num_wp = base._waypoints.shape[0]
    max_laps = base.cfg.max_n_laps
    # completion condition from env: (ng - 1) // num_wp >= max_laps  =>  ng >= num_wp*max_laps + 1
    target_gates = num_wp * max_laps + 1
    max_steps = int(base.cfg.episode_length_s / dt_step) + 10

    finished = torch.zeros(num_trials, dtype=torch.bool, device=device)
    ep_len = torch.zeros(num_trials, dtype=torch.long, device=device)
    success = torch.zeros(num_trials, dtype=torch.bool, device=device)
    crashed = torch.zeros(num_trials, dtype=torch.bool, device=device)
    # Per-env max gates seen across the entire episode (robust against auto-reset zeroing).
    max_ng = torch.zeros(num_trials, dtype=torch.long, device=device)
    final_ng = torch.zeros(num_trials, dtype=torch.long, device=device)

    # Snapshot domain-randomized parameters sampled at the initial reset.
    # These stay constant during the first episode of each env (until auto-reset resamples them).
    dr_twr = base._thrust_to_weight.detach().clone()
    dr_k_aero = base._K_aero.detach().clone()          # [N, 3]  xy, xy, z
    dr_kp_omega = base._kp_omega.detach().clone()      # [N, 3]  rp, rp, y
    dr_ki_omega = base._ki_omega.detach().clone()
    dr_kd_omega = base._kd_omega.detach().clone()
    dr_tau_m = base._tau_m.detach().clone()            # [N, 4]  (all 4 motors equal per-env)
    # initial position sampled per env
    dr_init_pos = base._robot.data.root_link_pos_w.detach().clone()  # [N, 3]

    obs = env.get_observations()
    if hasattr(obs, "get"):
        obs = obs["policy"]

    step = 0
    while not bool(finished.all()) and step < max_steps:
        # snapshot BEFORE step; this catches the pre-reset gate count on done envs
        prev_ng = base._n_gates_passed.long().clone()
        max_ng = torch.where(finished, max_ng, torch.maximum(max_ng, prev_ng))

        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]
        step += 1

        # reset_* flags reflect what happened on this step (before next step's physics)
        term = base.reset_terminated.bool()
        timeout = base.reset_time_outs.bool()

        # also update max_ng with post-step value for envs that didn't reset
        post_ng = base._n_gates_passed.long()
        max_ng = torch.where(finished | dones.bool(), max_ng, torch.maximum(max_ng, post_ng))

        dones_b = dones.bool()
        just_done = dones_b & ~finished
        if bool(just_done.any()):
            # completion-timeout triggers with ng = target_gates; the step that triggered it
            # had prev_ng = target_gates - 1 and passed the final gate during this step.
            # So the best estimate for "final gates at done" is prev_ng + 1 when completed.
            completed_done = just_done & timeout & ~term & (prev_ng + 1 >= target_gates)
            success[completed_done] = True

            # everything else that just finished is a failure
            crash_mask = just_done & ~completed_done
            crashed[crash_mask] = True

            ep_len[just_done] = step
            # record pre-reset gate count
            final_ng_this = torch.where(completed_done, prev_ng + 1, prev_ng)
            final_ng[just_done] = final_ng_this[just_done]
            finished |= just_done

            n_done = int(finished.sum())
            n_succ = int(success.sum())
            n_fail = int(crashed.sum())
            print(f"[step {step:4d}] {n_done:3d}/{num_trials} done "
                  f"(success: {n_succ}, fail: {n_fail})")

    # sweep up unfinished (hit max_steps without any done event) as failures
    if not bool(finished.all()):
        unfinished = ~finished
        crashed[unfinished] = True
        ep_len[unfinished] = step
        final_ng[unfinished] = max_ng[unfinished]
        finished[:] = True

    n_success = int(success.sum())
    n_fail = int(crashed.sum())
    success_rate = n_success / num_trials
    times_s = ep_len.float() * dt_step

    print("\n" + "=" * 60)
    print(f"Run:           {args_cli.load_run}")
    print(f"Checkpoint:    {args_cli.checkpoint}")
    print(f"Trials:        {num_trials}")
    print(f"Target gates:  {target_gates}  ({num_wp} gates x {max_laps} laps + 1 for completion)")
    print("-" * 60)
    print(f"Success rate:  {n_success}/{num_trials} = {success_rate*100:.1f}%")
    print(f"Failures:      {n_fail}")
    print("-" * 60)
    if n_success > 0:
        s_times = times_s[success]
        print(f"Time (success only):  mean={s_times.mean().item():.3f}s  "
              f"std={s_times.std().item():.3f}s  "
              f"min={s_times.min().item():.3f}s  "
              f"max={s_times.max().item():.3f}s")
    if n_fail > 0:
        f_times = times_s[crashed]
        f_gates = final_ng[crashed]
        print(f"Time (failures):      mean={f_times.mean().item():.3f}s")
        print(f"Gates passed (fail):  mean={f_gates.float().mean().item():.2f}  "
              f"min={int(f_gates.min().item())}  max={int(f_gates.max().item())}")
    print("=" * 60)

    # detailed per-failure listing (console)
    if n_fail > 0:
        fail_ids = torch.where(crashed)[0].tolist()
        print("\nFailure details (gate 1-indexed = last gate successfully passed + 1):")
        header = (
            f"  {'trial':>5} {'t(s)':>6} {'gate':>5} "
            f"{'TWR':>5} {'K_xy':>8} {'K_z':>8} "
            f"{'kp_rp':>6} {'ki_rp':>6} {'kd_rp':>5} "
            f"{'kp_y':>6} {'ki_y':>6} {'kd_y':>5} "
            f"{'tau_m(ms)':>9} "
            f"{'init_x':>7} {'init_y':>7} {'init_z':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in fail_ids:
            t_i = times_s[i].item()
            g_i = int(final_ng[i].item())
            # "which gate failed" = next gate it was trying for (1-indexed)
            failing_gate = g_i + 1 if g_i < target_gates - 1 else g_i
            twr = dr_twr[i].item()
            kxy = dr_k_aero[i, 0].item()
            kz = dr_k_aero[i, 2].item()
            kp_rp = dr_kp_omega[i, 0].item()
            ki_rp = dr_ki_omega[i, 0].item()
            kd_rp = dr_kd_omega[i, 0].item()
            kp_y = dr_kp_omega[i, 2].item()
            ki_y = dr_ki_omega[i, 2].item()
            kd_y = dr_kd_omega[i, 2].item()
            tau_ms = dr_tau_m[i, 0].item() * 1000.0
            ix, iy, iz = dr_init_pos[i].tolist()
            print(
                f"  {i:>5d} {t_i:>6.2f} {failing_gate:>5d} "
                f"{twr:>5.2f} {kxy:>8.2e} {kz:>8.2e} "
                f"{kp_rp:>6.1f} {ki_rp:>6.1f} {kd_rp:>5.2f} "
                f"{kp_y:>6.1f} {ki_y:>6.1f} {kd_y:>5.2f} "
                f"{tau_ms:>9.2f} "
                f"{ix:>7.2f} {iy:>7.2f} {iz:>7.2f}"
            )

    # write per-trial CSV (includes DR parameters)
    eval_csv = os.path.join(log_dir, f"eval_{args_cli.checkpoint.replace('.pt', '')}.csv")
    with open(eval_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial", "success", "crashed", "time_s",
            "gates_passed", "failing_gate", "max_gates",
            "twr", "k_aero_xy", "k_aero_z",
            "kp_omega_rp", "ki_omega_rp", "kd_omega_rp",
            "kp_omega_y", "ki_omega_y", "kd_omega_y",
            "tau_m_s",
            "init_x", "init_y", "init_z",
        ])
        for i in range(num_trials):
            g_i = int(final_ng[i].item())
            failing_gate = (g_i + 1 if (bool(crashed[i]) and g_i < target_gates - 1) else g_i)
            ix, iy, iz = dr_init_pos[i].tolist()
            w.writerow([
                i,
                int(success[i].item()),
                int(crashed[i].item()),
                f"{times_s[i].item():.3f}",
                g_i,
                failing_gate,
                int(max_ng[i].item()),
                f"{dr_twr[i].item():.4f}",
                f"{dr_k_aero[i, 0].item():.4e}",
                f"{dr_k_aero[i, 2].item():.4e}",
                f"{dr_kp_omega[i, 0].item():.3f}",
                f"{dr_ki_omega[i, 0].item():.3f}",
                f"{dr_kd_omega[i, 0].item():.4f}",
                f"{dr_kp_omega[i, 2].item():.3f}",
                f"{dr_ki_omega[i, 2].item():.3f}",
                f"{dr_kd_omega[i, 2].item():.4f}",
                f"{dr_tau_m[i, 0].item():.5f}",
                f"{ix:.3f}", f"{iy:.3f}", f"{iz:.3f}",
            ])
    print(f"\n[INFO] Wrote per-trial results to {eval_csv}")

    # plot
    if not args_cli.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(
                f"Eval: {args_cli.load_run} / {args_cli.checkpoint}\n"
                f"Success {n_success}/{num_trials} = {success_rate*100:.1f}%",
                fontsize=12,
            )

            # 1. success/fail bar
            ax = axes[0, 0]
            ax.bar(["success", "failure"], [n_success, n_fail], color=["tab:green", "tab:red"])
            ax.set_ylabel("count")
            ax.set_title("Outcome")
            for i, v in enumerate([n_success, n_fail]):
                ax.text(i, v + 0.5, str(v), ha="center")

            # 2. time histogram for successes
            ax = axes[0, 1]
            if n_success > 0:
                ax.hist(times_s[success].cpu().numpy(), bins=20, color="tab:green", edgecolor="k")
                ax.set_xlabel("completion time (s)")
                ax.set_ylabel("count")
                ax.set_title(f"Success times (n={n_success})")
            else:
                ax.text(0.5, 0.5, "no successes", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Success times")

            # 3. gates-passed histogram for failures
            ax = axes[1, 0]
            if n_fail > 0:
                fail_gates = final_ng[crashed].cpu().numpy()
                bins = max(1, int(fail_gates.max()) + 2)
                ax.hist(fail_gates, bins=range(bins + 1), color="tab:red", edgecolor="k", align="left")
                ax.set_xlabel("gates passed before failure")
                ax.set_ylabel("count")
                ax.set_title(f"Failure progression (n={n_fail})")
                ax.axvline(target_gates - 1, color="k", linestyle="--", alpha=0.5,
                           label=f"target={target_gates-1}")
                ax.legend()
            else:
                ax.set_title("Failure progression")

            # 4. per-trial scatter: time vs gates passed
            ax = axes[1, 1]
            t_cpu = times_s.cpu().numpy()
            g_cpu = final_ng.cpu().numpy()
            s_cpu = success.cpu().numpy()
            ax.scatter(t_cpu[~s_cpu], g_cpu[~s_cpu], c="tab:red", s=20, alpha=0.6, label="fail")
            ax.scatter(t_cpu[s_cpu], g_cpu[s_cpu], c="tab:green", s=20, alpha=0.6, label="success")
            ax.set_xlabel("episode time (s)")
            ax.set_ylabel("final gates passed")
            ax.axhline(target_gates - 1, color="k", linestyle="--", alpha=0.5)
            ax.set_title("Per-trial outcome")
            ax.legend()

            fig.tight_layout()
            out_path = args_cli.plot_out or os.path.join(
                log_dir, f"eval_{args_cli.checkpoint.replace('.pt', '')}.png"
            )
            fig.savefig(out_path, dpi=120)
            print(f"[INFO] Saved plot to {out_path}")
        except Exception as e:
            print(f"[WARN] plot failed: {e}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
